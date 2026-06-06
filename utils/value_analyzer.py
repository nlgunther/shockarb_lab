import argparse
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

def parse_value_report(file_path):
    # Read using latin-1 to seamlessly handle characters like 'é'
    with open(file_path, 'r', encoding='latin-1') as f:
        raw_text = f.read()

    # Clean up PDF/OCR artifacts and line breaks
    text = re.sub(r'\\', ' ', raw_text)
    text = re.sub(r'Research Highlights \|.*?Page \d+ of \d+', ' ', text, flags=re.DOTALL)
    text = re.sub(r'Source: Morningstar.*?2026', ' ', text, flags=re.DOTALL)
    text = re.sub(r'Stocks on Sale:.*?Market Cap \(B\)', ' ', text, flags=re.DOTALL)
    text = text.replace('\n', ' ')

    # Regex to capture the specific Morningstar row structure
    pattern = re.compile(
        r'(?P<Company>[A-Za-z0-9\s\&\.\,\'\-\(\)]+?)\s+'
        r'(?P<Stars>Q{4,5})\s+'
        r'(?P<Cur1>[A-Z]{3})\s+'
        r'(?P<FairValue>[\d\,]+(?:\.\d+)?)\s+'
        r'(?P<Cur2>[A-Z]{3})\s+'
        r'(?P<CurrentPrice>[\d\,]+(?:\.\d+)?)\s+'
        r'(?P<PFV>0\.\d+)\s+'
        r'(?P<Uncertainty>Low|Medium|High)\s+'
        r'(?P<Moat>Wide)\s+'
        r'(?P<MarketCap>[\d\,]+(?:\.\d+)?)'
    )

    data = []
    for match in pattern.finditer(text):
        row = match.groupdict()
        row['Company'] = row['Company'].strip()
        data.append(row)

    df = pd.DataFrame(data)
    df['PFV'] = df['PFV'].astype(float)
    df['MarketCap'] = df['MarketCap'].str.replace(',', '').astype(float)
    
    return df

def calculate_conviction(df):
    df['Discount'] = 1.0 - df['PFV']
    
    uncertainty_map = {'Low': 1.0, 'Medium': 1.5, 'High': 2.0}
    df['Uncertainty_Penalty'] = df['Uncertainty'].map(uncertainty_map)
    
    # Base calculation
    df['Log10_MarketCap'] = np.log10(df['MarketCap'])
    df['Conviction_Score'] = (df['Discount'] * df['Log10_MarketCap']) / df['Uncertainty_Penalty']
    
    return df.sort_values(by='Conviction_Score', ascending=False).reset_index(drop=True)

def get_pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    """Calculates the Pareto frontier for a given 2D set of points."""
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    return np.array(pareto_front)

def get_convex_frontier(Xs, Ys):
    """
    Calculates the upper convex hull (concave downward) using a pure Python 
    implementation of the Monotone Chain algorithm.
    """
    # Combine and sort points purely by X ascending
    points = sorted(zip(Xs, Ys))
    
    upper = []
    for p in points:
        while len(upper) >= 2:
            p1, p2 = upper[-2], upper[-1]
            # Cross product checks if the 3 points make a "left" or "right" turn.
            # We want the boundary to bulge UPWARDS, meaning strict right turns (cross < 0).
            cross = (p2[0] - p1[0]) * (p[1] - p2[1]) - (p2[1] - p1[1]) * (p[0] - p2[0])
            
            # If cross >= 0, the curve dips inward (concave), so we drop the middle point
            if cross >= 0:
                upper.pop()
            else:
                break
        upper.append(p)
        
    return np.array(upper)

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Calculates the shortest distance from a point to a line segment."""
    line_mag = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if line_mag == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Calculate the projection (dot product) to find the closest point on the line
    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
    u = max(min(u, 1.0), 0.0) # Clamp to the segment bounds
    
    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    return np.sqrt((px - ix)**2 + (py - iy)**2)

def calculate_distances_to_frontier(df, frontier_points, x_col, y_col):
    """Measures how far each stock is from the efficient frontier."""
    distances = []
    for _, row in df.iterrows():
        px, py = row[x_col], row[y_col]
        min_dist = float('inf')
        
        # Iterate through all segments of the convex hull
        for i in range(len(frontier_points) - 1):
            x1, y1 = frontier_points[i]
            x2, y2 = frontier_points[i+1]
            dist = point_to_segment_distance(px, py, x1, y1, x2, y2)
            min_dist = min(min_dist, dist)
            
        distances.append(min_dist)
    return distances

def plot_frontier(df, factor_col='Defense_Beta', save_path=None):
    plt.figure(figsize=(12, 8), facecolor='black')
    plt.style.use('dark_background')

    # ==========================================
    # 1. FACTOR MATH (The Hidden Dimension)
    # ==========================================
    factor_convex = get_convex_frontier(df[factor_col].tolist(), df['Discount'].tolist())
    df['Factor_Distance'] = calculate_distances_to_frontier(df, factor_convex, factor_col, 'Discount')
    
    max_dist = df['Factor_Distance'].max()
    sizes = 150 * (1 - (df['Factor_Distance'] / max_dist)) + 20 

    # ==========================================
    # 2. PLOTTING (The Conviction vs. Value Plane)
    # ==========================================
    pareto = get_pareto_frontier(df['Conviction_Score'].tolist(), df['Discount'].tolist(), maxX=True, maxY=True)
    plt.plot(pareto[:, 0], pareto[:, 1], color='lime', linestyle='--', linewidth=2, marker='s', markersize=6, alpha=1.0, zorder=2, label='Conviction Pareto Frontier')

    conviction_convex = get_convex_frontier(df['Conviction_Score'].tolist(), df['Discount'].tolist())
    plt.plot(conviction_convex[:, 0], conviction_convex[:, 1], color='white', linestyle='-', linewidth=1, alpha=0.5, label='Conviction Convex Frontier')

    scatter = plt.scatter(
        df['Conviction_Score'], df['Discount'], 
        c=df['Factor_Distance'], 
        s=sizes, 
        cmap='cool',   
        alpha=0.9,
        edgecolor='black',
        zorder=3
    )

    # ==========================================
    # 3. LABELING: THE SINGLE DEDUPLICATED SET
    # ==========================================
    frontier_coords = set()
    for pt in pareto:
        frontier_coords.add((round(pt[0], 6), round(pt[1], 6)))
    for pt in conviction_convex: 
        frontier_coords.add((round(pt[0], 6), round(pt[1], 6)))

    top_factor_companies = set(df.nsmallest(10, 'Factor_Distance')['Company'])

    points_to_annotate = {}
    
    for i, row in df.iterrows():
        x_val = round(row['Conviction_Score'], 6)
        y_val = round(row['Discount'], 6)
        company = row['Company']
        
        is_frontier = (x_val, y_val) in frontier_coords
        is_top_factor = company in top_factor_companies
        
        if is_frontier or is_top_factor:
            if company not in points_to_annotate or is_frontier:
                points_to_annotate[company] = (row['Conviction_Score'], row['Discount'], is_frontier)

    # EXACTLY ONE ANNOTATION LOOP
    for company, (x, y, is_frontier) in points_to_annotate.items():
        plt.annotate(
            company, 
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            color='white',
            fontweight='bold' if is_frontier else 'normal', 
            zorder=4
        )

    # ==========================================
    # 4. EXPLICIT FORMATTING & LABELING
    # ==========================================
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'Inefficiency (Distance to {factor_col} Frontier)')
    
    plt.title(f'Value Efficient Frontier (Conviction vs. Discount)\nMarkers Encoded by {factor_col} Efficiency')
    plt.xlabel('Conviction Score')
    plt.ylabel('Discount to Fair Value (1 - P/FV)')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    
    if save_path and save_path != 'show':
        plt.savefig(save_path, facecolor='black', edgecolor='black')
    else:
        plt.gcf().patch.set_facecolor('black')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and analyze wide-moat valuation data.")
    parser.add_argument('file', type=str, help="Path to the value screener text file")
    parser.add_argument('-g', '--graph', type=str, nargs='?', const='show', help="Plot the 2D efficient frontier. Provide a filename to save, or omit to display.")
    parser.add_argument('-o', '--output', type=str, help="Output file path for the CSV (e.g., results.csv)")

    args = parser.parse_args()

    try:
        raw_df = parse_value_report(args.file)
        ranked_df = calculate_conviction(raw_df)

        try:
            factors_df = pd.read_csv('shockarb_factors.csv')
            ranked_df = pd.merge(ranked_df, factors_df, on='Company', how='left')
            ranked_df['Defense_Beta'] = ranked_df['Defense_Beta'].fillna(0)
        except FileNotFoundError:
            print("\n[!] 'shockarb_factors.csv' not found. Injecting mock Defense_Beta for testing...")
            ranked_df['Defense_Beta'] = np.random.uniform(0, 0.5, size=len(ranked_df))
            ranked_df.loc[ranked_df['Company'].str.contains('Rheinmetall'), 'Defense_Beta'] = 1.85
            ranked_df.loc[ranked_df['Company'].str.contains('BAE Systems'), 'Defense_Beta'] = 1.70
            ranked_df.loc[ranked_df['Company'].str.contains('General Dynamics'), 'Defense_Beta'] = 1.65

        display_cols = ['Company', 'PFV', 'Discount', 'Uncertainty', 'MarketCap', 'Conviction_Score']
        print("\n⚡ SHOCKARB VALUE SCREENER: TOP 15 WIDE-MOAT TARGETS\n")
        print(ranked_df[display_cols].head(15).to_string(index=False))

        if args.output:
            ranked_df.to_csv(args.output, index=False)
            print(f"\n[+] Successfully exported full dataset to {args.output}")

        if args.graph:
            plot_frontier(ranked_df, factor_col='Defense_Beta', save_path=args.graph)
            if args.graph != 'show':
                print(f"[+] Successfully saved efficient frontier graph to {args.graph}")

    except FileNotFoundError:
        print(f"Error: Could not find '{args.file}'. Please verify the file path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
