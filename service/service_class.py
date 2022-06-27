import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


class Service:

    def __init__(self):
        self.data = pd.read_csv('../habits.csv')

    def give_recommendations(self, description, tags, index):
        result_list = []
        goals_vector = []
        goals_vector.append(description + "; " + tags)
        for el in self.data.values:
            goals_vector.append(el[0] + "; " + el[1])

        X = np.array(goals_vector)

        self.data = self.data[['Description', 'Tags', 'Recurrence']]
        self.data.head()

        self.data = self.data.dropna()

        text_data = X
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        embeddings = model.encode(text_data, show_progress_bar=True)

        embed_data = embeddings

        X = np.array(embed_data)
        n_comp = 5
        pca = PCA(n_components=n_comp)
        pca.fit(X)

        cos_sim_data = pd.DataFrame(cosine_similarity(X))
        index_recomm = cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:6]
        index2 = []
        for i in index_recomm:
            index2.append(i - 1)
        index_recomm = index2
        goals_recommendations = self.data['Description'].loc[index_recomm].values

        k = 1
        for q in range(len(goals_recommendations)):
            plot_desc = self.data['Description'].loc[index_recomm[q]]

            plot_tags = self.data['Tags'].loc[index_recomm[q]]

            plot_recurrence = self.data['Recurrence'].loc[index_recomm[q]]
            recommendation = {"order": k, "description": plot_desc, "tags": plot_tags,
                              "recurrence": plot_recurrence}
            result_list.append(recommendation)
            k = k + 1

        return result_list
