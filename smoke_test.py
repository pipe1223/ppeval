from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from evaluation.evaluation import evaluate

rag_path = root / 'evaluation' / 'retrieval' / 'examples' / 'rag_sample.json'
graph_path = root / 'evaluation' / 'retrieval' / 'examples' / 'graphrag_sample.json'

print('RAG:')
print(json.dumps(evaluate(rag_path, mode='rag'), indent=2))
print('GraphRAG:')
print(json.dumps(evaluate(graph_path, mode='graphrag'), indent=2))
