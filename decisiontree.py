import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics

import pydot

data = pd.read_csv('weather.csv')
df = pd.DataFrame(data=data)

# remove unimportant data
df = df.replace('haze', np.nan)
df = df.replace('smoke', np.nan)
df = df.replace('volcanic ash', np.nan)
df = df.replace('fog', np.nan)
df = df.dropna(how='any')

# simplify output classes
df = df.replace('mist', 'light rain')
df = df.replace('broken clouds', 'few clouds')
df = df.replace('heavy intensity drizzle', 'drizzle')
df = df.replace('light intensity drizzle', 'drizzle')
df = df.replace('heavy intensity rain', 'heavy rain')
df = df.replace('very heavy rain', 'heavy rain')
df = df.replace('thunderstorm with heavy rain', 'thunderstorm')
df = df.replace('thunderstorm with light rain', 'thunderstorm')
df = df.replace('thunderstorm with rain', 'thunderstorm')
df = df.replace('light rain and snow', 'light rain')
df = df.replace('moderate rain', 'rain')
df = df.replace('light rain', 'rain')
df = df.replace('overcast clouds', 'cloudy')
df = df.replace('scattered clouds', 'cloudy')
df = df.replace('proximity thunderstorm', 'thunderstorm')
df = df.replace('squalls', 'snow')
df = df.replace('few clouds', 'sky is clear')
df = df.replace('heavy snow', 'snow')
df = df.replace('light snow', 'snow')
df = df.replace('heavy rain', 'rain')
df = df.replace('drizzle', 'rain')

train = df.head(40000)
test = df.tail(5252)

weather = train['Weather']
features = train.drop(['Weather'], axis=1)

test_weather = test['Weather']
test_features = test.drop(['Weather'], axis=1)

le = preprocessing.LabelEncoder()

weather_encoded = le.fit_transform(weather)




clf = tree.DecisionTreeClassifier()
clf.fit(features, weather_encoded)

pred = le.inverse_transform(clf.predict(test_features))
print(metrics.accuracy_score(y_pred=pred, y_true=test_weather))

