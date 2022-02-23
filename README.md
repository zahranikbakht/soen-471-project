Introduction
“Trust me! You have to see this movie” is a great way to convince your friends about a new cinema endeavor, but what truly matters for a movie to be a hit or a flop? We plan to create a model that predicts the success of a movie based on a number of features. The main question that we aim to answer in this project is what are the significant factors that affect the profitability of a movie. As such, our training sets would include movies released in theaters from the past three decades along with features we will list in the datasets given. In addition to trying to find what significant factors are affecting movie profitability, our trained model can then be used to predict how successful a new movie entered into the model would be.  
 
Dataset
We plan to combine 3 datasets to get various information on the 1000 top grossing movies. Initially, we will take the dataset found here which includes data on the top 1000 highest grossing movies to date. It will be merged with another dataset from here which includes data on 5000 movies from TMDB and another one here that has 50000 movies from IMDB to provide more features for our model.
The 14 features that we are going to incorporate in our model are: Distributor Name, Genre, Runtime, Release Date, Budget, Number of the cast, gender of the main actor, gender of the second-main actor, Number of the crew, TMBD rating, IMDB rating, content rating (PG-13, R, etc), total cast Facebook likes and movie Facebook likes.
The range of movie revenue that we have in our database is from 81 million dollars to 2.8 billion dollars. We will categorize this range into 10 equal-size intervals of length 200 million dollars each. We will then predict which of these 10 intervals (labels) a movie will fall into. It’s important to note that we aim to provide an estimate of a movie’s success rather than its precise revenue.
 
Model
First, we preprocess and combine the 3 datasets using Spark. We will be using supervised classification for our model, since all our data is clearly labeled and we have a classification task at hand. The two algorithms that we are going to use and compare are Random Forest and Logistic Regression with Apache Spark. RF allows us to recognize the most discriminating features in our model to answer our research question. LR would allow us to estimate the probability of the event occurring with a target variable in an efficient way.


## 🧑‍💻 Team members
| Names  | Student ID | GitHub handle | 
| ------------- | ------------- | ------------- |
| Rami Rafeh            | 29198024   | [ramirafeh](https://github.com/ramirafeh)|
| Noah-James Dinh       | 40128079   | [eyeshield2110](https://github.com/eyeshield2110)  |
| Zahra Nikbakht        | 40138253   | [zahranikbakht](https://github.com/zahranikbakht)  |
| Vaansh Vikas Lakhwara | 40114764   | [vaansh](https://github.com/vaansh)  |
