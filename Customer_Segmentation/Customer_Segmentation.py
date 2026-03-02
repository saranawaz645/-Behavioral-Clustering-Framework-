import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class MarketSegmentAnalyzer:
    """
    Advanced system for identifying customer behavior patterns using K-Means.
    Features: Automated preprocessing, Elbow analysis, and high-fidelity plotting.
    """
    
    def __init__(self, csv_file):
        """Setup analyzer with dataset path and normalization tools."""
        self.csv_file = csv_file
        self.raw_data = None
        self.scaled_features = None
        self.model = None
        self.normalization_tool = StandardScaler()
        self.cluster_centers = None
        self.segment_labels = {}

    def prepare_data(self):
        """Step 1: Data Cleaning and Standardization."""
        try:
            self.raw_data = pd.read_csv(self.csv_file)
            # Isolating relevant metrics: Income and Spending behavior
            data_points = self.raw_data.iloc[:, [3, 4]].values
            
            # Standardization: Crucial for accurate distance calculation in KMeans
            self.scaled_features = self.normalization_tool.fit_transform(data_points)
            print("[OK] Data ingestion and normalization complete.")
        except Exception as error:
            print(f"[ERR] Data load failed: {error}")

    def execute_elbow_test(self):
        """Step 2: Cluster Optimization via WCSS (Inertia) analysis."""
        inertia_values = []
        for n in range(1, 11):
            km_test = KMeans(n_clusters=n, init='k-means++', random_state=42)
            km_test.fit(self.scaled_features)
            inertia_values.append(km_test.inertia_)
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), inertia_values, marker='D', color='#1a5276', ls='-')
        plt.title('Optimization: Elbow Curve for Cluster Selection', fontsize=13)
        plt.xlabel('Cluster Count (K)')
        plt.ylabel('WCSS (Metric of Cohesion)')
        plt.grid(True, alpha=0.3)
        plt.savefig('Elbow_Optimization_Plot.png')
        print("[FILE] Elbow curve saved as 'Elbow_Optimization_Plot.png'.")

    def run_clustering(self, n_segments=5):
        """Step 3: Executing the final K-Means algorithm."""
        self.model = KMeans(n_clusters=n_segments, init='k-means++', random_state=42)
        self.raw_data['Segment_ID'] = self.model.fit_predict(self.scaled_features)
        
        # Reverting scaling for interpretability of cluster centers
        self.cluster_centers = self.normalization_tool.inverse_transform(self.model.cluster_centers_)
        self._assign_segment_names()
        print(f"[OK] Model successfully partitioned data into {n_segments} groups.")

    def _assign_segment_names(self):
        """Internal Mapping: Defining personas based on spending/income data."""
        for idx, (income, spend) in enumerate(self.cluster_centers):
            if income > 70 and spend > 70:
                self.segment_labels[idx] = {'title': 'Elite/VIP', 'hex': '#27ae60'} 
            elif income > 70 and spend < 40:
                self.segment_labels[idx] = {'title': 'Conservative', 'hex': '#c0392b'}
            elif income < 40 and spend > 70:
                self.segment_labels[idx] = {'title': 'Impulsive', 'hex': '#8e44ad'}
            elif income < 40 and spend < 40:
                self.segment_labels[idx] = {'title': 'Frugal', 'hex': '#f39c12'}
            else:
                self.segment_labels[idx] = {'title': 'Moderate', 'hex': '#2980b9'}

    def export_statistical_report(self):
        """Step 4: Generating a numerical breakdown for business reporting."""
        print("\n" + "*"*70)
        print("                 CUSTOMER ANALYTICS: SEGMENT BREAKDOWN")
        print("*"*70)
        print(f"{'Persona Type':<18} | {'Total':<8} | {'Mean Income':<15} | {'Mean Spend'}")
        print("-" * 70)
        
        report_log = []
        for idx in range(len(self.cluster_centers)):
            group = self.raw_data[self.raw_data['Segment_ID'] == idx]
            name = self.segment_labels[idx]['title']
            pop = len(group)
            m_inc = group.iloc[:, 3].mean()
            m_spd = group.iloc[:, 4].mean()
            
            print(f"{name:<18} | {pop:<8} | ${m_inc:<14.2f} | {m_spd:.2f}")
            report_log.append([name, pop, m_inc, m_spd])
        
        # Exporting statistics to CSV
        summary_export = pd.DataFrame(report_log, columns=['Persona', 'Count', 'Avg_Income', 'Avg_Spending'])
        summary_export.to_csv('Market_Segment_Report.csv', index=False)
        print("*"*70)
        print("[FILE] Statistics exported to 'Market_Segment_Report.csv'.")

    def plot_market_segments(self):
        """Step 5: Visual representation of segmented customer base."""
        plt.figure(figsize=(12, 8))
        
        for idx in range(len(self.cluster_centers)):
            data_subset = self.raw_data[self.raw_data['Segment_ID'] == idx]
            plt.scatter(data_subset.iloc[:, 3], data_subset.iloc[:, 4], 
                        s=90, c=self.segment_labels[idx]['hex'], 
                        label=self.segment_labels[idx]['title'], edgecolors='white', linewidth=0.5)

        # Highlighting centers as large 'Diamond' markers
        plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], s=300, 
                    c='black', marker='D', label='Segment Centroids', edgecolors='white')
        
        plt.title('Visualization: Customer Behavioral Segments', fontsize=15)
        plt.xlabel('Annual Earnings (k$)', fontsize=11)
        plt.ylabel('Spending Index (1-100)', fontsize=11)
        plt.legend(title="Customer Persona", loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('Final_Segmentation_Map.png')
        print("[FILE] Cluster visualization saved as 'Final_Segmentation_Map.png'.")
        plt.show()

# --- Main Driver Script ---
if __name__ == "__main__":
    # Initializing the Clustering Engine
    processor = MarketSegmentAnalyzer('Mall_Customers.csv')
    
    # Process Pipeline
    processor.prepare_data()
    processor.execute_elbow_test()
    processor.run_clustering(n_segments=5)
    processor.export_statistical_report()
    processor.plot_market_segments()
    
    print("\n[COMPLETE] Analysis pipeline finished successfully.")