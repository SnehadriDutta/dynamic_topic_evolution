import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Helper:

    @staticmethod
    def read_file(file_path: str):
        with open(file_path, 'r') as file:
            read_json = json.load(file)
        return read_json

    @staticmethod
    def extract_data(file_paths: list[str]):
        dfs = []
        for i, path in enumerate(file_paths):
            data = Helper.read_file(file_path=path)
            p_rows = []
            t_rows = []
            for post in data['posts']:
                p_rows.append({
                        'id': post['id'],
                        f'run_{i + 1}': str(int(post['topic']))
                    })
            for topic in data['topic_entities']:
                t_rows.append({
                    'topic': str(int(topic['topic'])),
                    'title': topic['title'],
                    'summary': topic['summary'],
                    'key_entities': topic['key_entities']
                })
            dfs.append((pd.DataFrame(p_rows), pd.DataFrame(t_rows)))
        return dfs
    
    @staticmethod
    def clean_df(df1, df2, df3, merge_on, merge_how, run_cols):
        post_df = df1.merge(df2, on=merge_on, how=merge_how).merge(df3, on=merge_on, how=merge_how)
        post_df = post_df.fillna(-1)
        valid_count = post_df[run_cols].apply(lambda row: ((row != np.nan) & (row != -1)).sum(), axis=1)
        post_df = post_df[valid_count >= 2].replace({-1: '-1'})
        return post_df

    @staticmethod
    def get_entity_changes(prior_run_entity, later_run_entity):
            s1 = set([e[0] for e in prior_run_entity] if isinstance(prior_run_entity, list) else [])
            s2 = set([e[0] for e in later_run_entity] if isinstance(later_run_entity, list) else [])

            return {
                'added_entities': list(s2-s1),
                'removed_entities': list(s1-s2)
            }

    @staticmethod
    def fetch_title_summary(row):
            title = str(row.get('title', ''))

            summary = row.get('summary', '')
            if isinstance(summary, list):
                summary = ' '.join(summary)

            return f"{title} {summary}"
    
    @staticmethod
    def categorize_evolution(jaccard_score):
        """Generates a human-readable label for the UI/Stakeholders."""
        if jaccard_score >= 0.8: 
            return "Stable"
        elif jaccard_score >= 0.5: 
            return "Evolved"
        elif jaccard_score > 0.1: 
            return "Fragmented"
        return "Dropped/Noise"

class TopicEvolutionPipeline:
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def run_comparison(self, post_df: pd.DataFrame, t_prior: pd.DataFrame, t_later: pd.DataFrame, 
                       post_id_col:str, prior_run_col: str, later_run_col: str):
        
        p_run_sets = post_df.groupby(prior_run_col)[post_id_col].apply(set).to_dict()
        l_run_sets = post_df.groupby(later_run_col)[post_id_col].apply(set).to_dict()

        t_prior['combined_text'] = t_prior.apply(Helper.fetch_title_summary, axis=1)
        t_later['combined_text'] = t_later.apply(Helper.fetch_title_summary, axis=1)

        meta_prior = t_prior.set_index('topic').to_dict('index')
        meta_later = t_later.set_index('topic').to_dict('index')

        results = []

        for tp, post_p in p_run_sets.items():
            for tl, post_l in l_run_sets.items():

                intersection = len(post_p.intersection(post_l))

                if intersection == 0:
                    continue

                union = len(post_p.union(post_l))
                jaccard = intersection / union

                text_p = meta_prior.get(tp, {}).get('combined_text', '')
                text_l = meta_later.get(tl, {}).get('combined_text', '')

                cos_sim =0.0
                if text_p and text_l:
                    tfidf_matrix = self.vectorizer.fit_transform([text_p, text_l])
                    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


                ents_p = meta_prior.get(tp, {}).get('key_entities', [])
                ents_l = meta_later.get(tl, {}).get('key_entities', [])
                ent_diff = Helper.get_entity_changes(ents_p, ents_l)

                results.append({
                    'Earlier Topic': tp,
                    'New Topic': tl,
                    'Earlier Title': meta_prior.get(tp, {}).get('title', ''),
                    'New Title': meta_later.get(tl, {}).get('title', ''),
                    'Evolution Status': Helper.categorize_evolution(jaccard),
                    'Jaccard Score': round(jaccard, 3),
                    'Cosine Similarity': round(cos_sim, 3),
                    'Added Entities': ent_diff['added_entities'],
                    'Removed Entities': ent_diff['removed_entities']
                })
        return pd.DataFrame(results)

    def create_post_level_timeline(self, post_df, df_1_to_2, df_2_to_3):
        
        p_df = post_df.copy()
        run_cols = ['run_1', 'run_2', 'run_3']
        p_df[run_cols] = p_df[run_cols].fillna("-1").astype(str)

        
        hop1_metrics = df_1_to_2[['Earlier Topic', 'New Topic', 'Evolution Status', 'Jaccard Score']].copy()
        hop1_metrics = hop1_metrics.rename(columns={
            'Evolution Status': 'Run 1->2 Status',
            'Jaccard Score': 'Run 1->2 Score'
        })

        
        hop2_metrics = df_2_to_3[['Earlier Topic', 'New Topic', 'Evolution Status', 'Jaccard Score']].copy()
        hop2_metrics = hop2_metrics.rename(columns={
            'Evolution Status': 'Run 2->3 Status',
            'Jaccard Score': 'Run 2->3 Score'
        })

        post_timeline = pd.merge(
            p_df, 
            hop1_metrics, 
            left_on=['run_1', 'run_2'], 
            right_on=['Earlier Topic', 'New Topic'], 
            how='left'
        )
        
        post_timeline = pd.merge(
            post_timeline, 
            hop2_metrics, 
            left_on=['run_2', 'run_3'], 
            right_on=['Earlier Topic', 'New Topic'], 
            how='left'
        )

        
        post_timeline = post_timeline.drop(columns=['topic_run1_id', 'topic_run2_id_x', 'topic_run2_id_y', 'topic_run3_id'], errors='ignore')
        
        post_timeline['Run 1->2 Status'] = post_timeline['Run 1->2 Status'].fillna("Outlier / Dropped")
        post_timeline['Run 2->3 Status'] = post_timeline['Run 2->3 Status'].fillna("Outlier / Dropped")
        post_timeline['Run 1->2 Score'] = post_timeline['Run 1->2 Score'].fillna(0.0)
        post_timeline['Run 2->3 Score'] = post_timeline['Run 2->3 Score'].fillna(0.0)

        final_cols = ['id', 'run_1', 'run_2', 'Run 1->2 Status', 'Run 1->2 Score', 'run_3', 'Run 2->3 Status', 'Run 2->3 Score']
        
        available_cols = [c for c in final_cols if c in post_timeline.columns]
        
        return post_timeline[available_cols]


def main():
    files = [r'Data\\2026-02-19_posts.json', r'Data\\2026-02-20_posts.json', r'Data\\2026-02-21_posts.json']
    
    dfs = Helper.extract_data(file_paths=files)

    p_df1, t_df1 = dfs[0]
    p_df2, t_df2 = dfs[1]
    p_df3, t_df3 = dfs[2]

    post_df = Helper.clean_df(p_df1, p_df2, p_df3, merge_how='outer', merge_on='id', run_cols=['run_1', 'run_2', 'run_3'])

    # Initialize the pipeline
    pipeline = TopicEvolutionPipeline()

    # 1. Measure changes from Run 1 (Feb 19) to Run 2 (Feb 20)
    print("Processing Run 1 to Run 2...")
    evolution_1_to_2 = pipeline.run_comparison(
        post_df=post_df,
        t_prior=t_df1,
        t_later=t_df2,
        post_id_col='id',
        prior_run_col='run_1',
        later_run_col='run_2')
    # Add a tracking column
    evolution_1_to_2['time_step'] = 'Feb 19 -> Feb 20'


    # 2. Measure changes from Run 2 (Feb 20) to Run 3 (Feb 21)
    print("Processing Run 2 to Run 3...")
    evolution_2_to_3 = pipeline.run_comparison(
        post_df=post_df,
        t_prior=t_df2,
        t_later=t_df3,
        post_id_col='id',
        prior_run_col='run_2',
        later_run_col='run_3')
    # Add a tracking column
    evolution_2_to_3['time_step'] = 'Feb 20 -> Feb 21'

    # 3. Combine into a master timeline
    master_timeline_df = pd.concat([evolution_1_to_2, evolution_2_to_3], ignore_index=True)
    master_timeline_df.to_csv('Topic_Comparison.csv', index=False)

    print("Processing Post Hop Run1 to Run 2 to Run 3...")
    post_timeline_df = pipeline.create_post_level_timeline(post_df, evolution_1_to_2, evolution_2_to_3)
    post_timeline_df.to_csv("Post_Comparison.csv", index=False)   


if __name__ == "__main__":
    main()







