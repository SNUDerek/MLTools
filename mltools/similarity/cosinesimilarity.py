import pandas as pd
from scipy.spatial.distance import cosine

class CosineSimRetriever:
    def __init__(self, vectorizer, questions, responses, q_vectors=None):

        self.vectorizer = vectorizer
        self.questions = questions
        self.responses = responses
        self.q_vectors = q_vectors

    def fit(self):
        """initalize the vectorizer and get question vectors"""
        self.vectorizer.fit(self.questions)
        self.q_vectors = self.vectorizer.predict(self.questions)

    def _get_feature_vector(self, sentence):
        """get single sentence vector"""
        feat_vect = self.vectorizer.predict(sentence)
        return feat_vect[0]

    def query(self, sentence, count=10):
        """query the database and return top (n) sentences"""
        feat_vect = self._get_feature_vector(sentence)

        results = []

        for index, vector in enumerate(self.q_vectors):

            if feat_vect is 0 or feat_vect is None or vector is None or vector is 0:
                continue
            cosine_sim = 1 - cosine(feat_vect, vector)
            if cosine_sim > 0.5:
                results.append((cosine_sim, self.questions[index], self.responses[index]))

        results = sorted(results, key=lambda x: x[0], reverse=True)

        return results[:count]

    def getdf(self, sentence, count=10):
        """get pandas DataFrame of top (n) examples"""
        results = self.query(sentence, count)

        sim_query = [sentence]
        sim_scores = []
        sim_questions = []
        sim_responses = []

        for result in results:
            sim_scores.append(result[0])
            sim_questions.append(result[1])
            sim_responses.append(result[2])

        for i in range(1, len(sim_responses)):
            sim_query.append(' ')

        minlen = min([len(sim_query), len(sim_scores),
                      len(sim_questions), len(sim_responses)])

        answer = pd.DataFrame(
            {'query': sim_query[:minlen],
             'request': sim_questions[:minlen],
             'response': sim_responses[:minlen],
             })

        answer = answer[['query', 'score', 'request', 'response']]

        return answer
