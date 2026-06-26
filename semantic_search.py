import json
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


#embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

class Helper:

    @staticmethod
    def read_file(file_path: str):
        with open(file_path, 'r') as file:
            read_json = json.load(file)
        return read_json

    @staticmethod
    def extract_data(file_paths: list[str]):
        results = []
        for i, path in enumerate(file_paths):
            data = Helper.read_file(file_path=path)            
            t_rows = []            
            for topic in data['topic_entities']:

                topic_id = topic['topic']

                if topic_id ==-1:
                    continue
                
                title = topic.get('title', '')
                summary = topic.get('summary', [])
                summary = ' '.join(summary)

                key_entities = topic.get('key_entities', [])
                primary_entities = [entity_list[0] for entity_list in key_entities if entity_list]
                entities = ", ".join(primary_entities)
                combination = f"Title: {title}; Summary: {summary}; entities: {entities}"
                
                text_for_matching = f"{title} {summary}"
                tokens = set(re.findall(r'\b\w+\b', text_for_matching))

                t_rows.append({
                    'topic': str(int(topic_id)),
                    'title': topic['title'],
                    'combination': combination,
                    'tokens': tokens
                })
            results.append(t_rows)
        return results
    
    @staticmethod
    def calculate_jaccard(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return round((intersection / union), 3)
    

class EmbedderNRetreiver:

    def __init__(self, jaccard_threshold=0.5):
        self.jaccard_threshold = jaccard_threshold
        self.unique_topics = []
        self.embeddings = None


    def compare_topics(self, raw_topics):
        
        for new_topic in raw_topics:
            is_duplicate = False

            for existing_topic in self.unique_topics:
                score = Helper.calculate_jaccard(existing_topic['tokens'], new_topic['tokens'])

                if score > self.jaccard_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                self.unique_topics.append(new_topic)

    def embed_topics(self):
        
        texts_to_embed = [topic['combination'] for topic in self.unique_topics]
        self.embeddings = embedder.encode(texts_to_embed, show_progress_bar=True)

        for i, topic in enumerate(self.unique_topics):
            topic['embedding'] = self.embeddings[i].tolist()

    def search(self, query):
        
        query_embedding = embedder.encode([query])

        similarity_score = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(similarity_score)[::-1][:5]

        results = []

        for i in top_indices:
            results.append({
                'topic': self.unique_topics[i]['topic'],
                'title': self.unique_topics[i]['title'],
                'similarity_score': round(similarity_score[i], 4)
            })
        return results
    

def main():
    files = [r'Data\\2026-02-19_posts.json', r'Data\\2026-02-20_posts.json', r'Data\\2026-02-21_posts.json']
    
    topics = Helper.extract_data(file_paths=files)

    embedder_retriever = EmbedderNRetreiver(0.5)

    print("Checking for Duplicate topics")
    for topics in topics:
        embedder_retriever.compare_topics(topics)

    print("Embedding the topics")
    embedder_retriever.embed_topics()

    while True:        
        print("="*50)
        print("\n")

        query = input("Write your query: ")
        results = embedder_retriever.search(query)
        
        for res in results:
            print(f"Topic: {res['topic']}")
            print(f'Title: {res['title']}')
            print(f'Similarity Score: {res['similarity_score']}')
            



if __name__ == "__main__":
    main()    
