import networkx as nx
import re
from collections import defaultdict
 
class LegalCitationGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.metadata = {}
 
    def add_judgment(self, case_id, text, year, bench_size=3):
        self.G.add_node(case_id, year=year, bench=bench_size)
        self.metadata[case_id] = {'text': text, 'year': year}
        cited = self.extract_citations(text)
        for c in cited:
            self.G.add_edge(case_id, c, type='cites')
 
    def extract_citations(self, text):
        pattern = r'AIR d{4} SC d+|(d{4}) d+ SCC d+|(d{4}) d+ SCR d+'
        return list(set(re.findall(r'AIR d{4} SC d+', text)))
 
    def pagerank_importance(self):
        return nx.pagerank(self.G, alpha=0.85)
 
    def find_landmark_cases(self, top_k=5):
        pr = self.pagerank_importance()
        in_deg = dict(self.G.in_degree())
        combined = {n: pr.get(n,0)*0.6 + in_deg.get(n,0)*0.4/max(in_deg.values(),default=1)
                    for n in self.G.nodes()}
        return sorted(combined, key=combined.get, reverse=True)[:top_k]
 
    def detect_overruled(self):
        overruled = []
        for node in self.G.nodes():
            successors = list(self.G.successors(node))
            if any('overrule' in self.metadata.get(s, {}).get('text','').lower()
                   for s in successors):
                overruled.append(node)
        return overruled
 
    def community_analysis(self):
        ug = self.G.to_undirected()
        if ug.number_of_edges() == 0:
            return []
        from networkx.algorithms.community import greedy_modularity_communities
        return list(greedy_modularity_communities(ug))
 
g = LegalCitationGraph()
cases = [
    ("CASE_001", "The constitution bench held... AIR 1973 SC 1461 is affirmed.", 2010),
    ("CASE_002", "Following AIR 1973 SC 1461 and CASE_001 precedent...", 2015),
    ("CASE_003", "Distinguished from CASE_002, we hold...", 2018),
    ("CASE_004", "AIR 1973 SC 1461 remains binding. CASE_001 applies.", 2020),
]
for cid, text, year in cases:
    g.add_judgment(cid, text, year)
print("Nodes:", g.G.number_of_nodes(), "Edges:", g.G.number_of_edges())
print("Landmark cases:", g.find_landmark_cases())
pr = g.pagerank_importance()
for k,v in sorted(pr.items(), key=lambda x:-x[1]):
    print(f"  {k}: PageRank={v:.3f}")
