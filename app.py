from flask import Flask, request, jsonify , json
import pickle
import regex as re
import numpy as np
from flashtext import KeywordProcessor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

input_data = pickle.load(open('input_data.pkl','rb'))
symptoms = pickle.load(open('symptoms.pkl','rb'))

x = input_data.drop(['prognosis'],axis =1)
y = input_data['prognosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)



gbm_clf = GradientBoostingClassifier()
gbm_clf.fit(x_train, y_train)

features = input_data.columns[:-1]

feature_dict = {}
for i,f in enumerate(features):
    feature_dict[f] = i

# sample = [i/52 if i ==52 else i/24 if i==24 else i*0 for i in range(len(features))]

# sample_x = np.array(sample).reshape(1,len(sample))

keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(symptoms)

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
        query = request.form.get('query')

        matched_keyword = keyword_processor.extract_keywords(query)
        if len(matched_keyword) == 0:
            return jsonify({"No Matches"})
        else:
            regex = re.compile(' ')
            processed_keywords = [i if regex.search(i) == None else i.replace(' ', '_') for i in matched_keyword]
            #print(processed_keywords)
            coded_features = []
            for keyword in processed_keywords:
                coded_features.append(feature_dict[keyword])
            print(coded_features)
            sample_x = []
            for i in range(len(features)):
                try:
                    sample_x.append(i / coded_features[coded_features.index(i)])
                except:
                    sample_x.append(i * 0)
            sample_x = np.array(sample_x).reshape(1, len(sample_x))
            Predicted_Disease = gbm_clf.predict(sample_x)[0]
            ans = json.dumps({'Disease': Predicted_Disease})

            return jsonify(ans)

if __name__== '__main__':
    app.run(host="0.0.0.0", port=5000)