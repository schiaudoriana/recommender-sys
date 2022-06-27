import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA



plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta',
          'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
          'xkcd:scarlet']

data = pd.read_csv('../habits.csv')

vector_pls_work = []
description = "Drink 2L of smoothies"
tags = "diet, health"
recurrence = "daily"  # not really used anywhere tho
vector_pls_work.append(description)
for el in data.values:
    vector_pls_work.append(el[0])

X = np.array(vector_pls_work)
# X = np.array(data.Tags)

# X = np.array(data).flatten()

# print(data.columns)

data = data[['Description', 'Tags', 'Recurrence']]
data.head()

data = data.dropna()

text_data = X
# print(text_data)
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(text_data, show_progress_bar=True)

embed_data = embeddings

X = np.array(embed_data)
n_comp = 5
pca = PCA(n_components=n_comp)
pca.fit(X)
pca_data = pd.DataFrame(pca.transform(X))

print(pca_data.head())

sns.pairplot(pca_data)
# plt.show()

cos_sim_data = pd.DataFrame(cosine_similarity(X))


def give_recommendations(index, print_recommendation=False, print_recommendation_plots=False, print_genres=False):
    index_recomm = cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:6]
    movies_recomm = data['Description'].loc[index_recomm].values
    result = {'Goals': movies_recomm, 'Index': index_recomm}
    if print_recommendation:
        print('The given goal is this one: %s \n' % description)
        k = 1
        for movie in movies_recomm:
            print('The number %i recommended goal is this one: %s \n' % (k, movie))
    if print_recommendation_plots:
        print('The tags of the given goal are:\n %s \n' % tags)
        k = 1
        for q in range(len(movies_recomm)):
            plot_q = data['Tags'].loc[index_recomm[q]]
            print('The tags of the number %i recommended goals are:\n %s \n' % (k, plot_q))
            k = k + 1
    if print_genres:
        print('The recurrence of the given goal is this one:\n %s \n' % recurrence)
        k = 1
        for q in range(len(movies_recomm)):
            plot_q = data['Recurrence'].loc[index_recomm[q]]
            print('The recurrence of the number %i recommended goal is this one:\n %s \n' % (k, plot_q))
            k = k + 1
    return result


plt.figure(figsize=(20, 20))
for q in range(1, 5):
    plt.subplot(2, 2, q)
    index = np.random.choice(np.arange(0, len(X)))
    to_plot_data = cos_sim_data.drop(index, axis=1)
    plt.plot(to_plot_data.loc[index], '.', color='firebrick')
    recomm_index = give_recommendations(index)
    x = recomm_index['Index']
    y = cos_sim_data.loc[index][x].tolist()
    m = recomm_index['Goals']
    plt.plot(x, y, '.', color='navy', label='Recommended Goals')
    plt.title('Goal given: ' + data['Description'].loc[index])
    plt.xlabel('Goal Index')
    k = 0
    for x_i in x:
        plt.annotate('%s' % (m[k]), (x_i, y[k]), fontsize=10)
        k = k + 1

    plt.ylabel('Cosine Similarity')
    plt.ylim(0, 1)

give_recommendations(0, True, True)
plt.show()
