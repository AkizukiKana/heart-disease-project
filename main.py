import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('weather.csv')
df = pd.DataFrame(data=data)

# remove unimportant data
df = df.replace('haze', np.nan)
df = df.replace('smoke', np.nan)
df = df.replace('volcanic ash', np.nan)
df = df.replace('fog', np.nan)
df = df.dropna(how='any')

# simplify the output classes
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


allWeather = df["Weather"]
allFeatures = df.drop('Weather', axis=1)

le = preprocessing.LabelEncoder()

weather_encoded = le.fit_transform(allWeather)
df['Weather'] = weather_encoded
allWeather = df["Weather"]

poly = PolynomialFeatures(degree=1, interaction_only=True)
polynomials = pd.DataFrame(poly \
                           .fit_transform(allFeatures))

allFeatures = polynomials

weather = allWeather.head(40000)
features = allFeatures.head(40000)

test_weather = allWeather.tail(5252)
test_features = allFeatures.tail(5252)
model = GaussianNB()

model.fit(features, weather)

class_pred = model.predict(test_features)

df = pd.DataFrame(data=class_pred)
df.to_csv('prediction.csv')

test_weather.to_csv('actual.csv')

print(metrics.accuracy_score(test_weather, class_pred))
