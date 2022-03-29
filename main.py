import csv
import os
import sys
# Spark imports
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, udf, col, split
import pandas as pd

from pyspark.sql.types import IntegerType, StringType


# Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def load_df_from_csv(filename):
    spark = init_spark()
    df = spark.read.csv(filename, header=True, multiLine=True, quote="\"", escape="\"")
    return df


def json_list_num(all_cast):
    cleaned = all_cast.replace("/", "")
    converted_list = list(eval(cleaned))
    return len(converted_list)


def gender_of_first_cast(all_cast):
    cleaned = all_cast.replace("/", "")
    converted_list = list(eval(cleaned))
    if converted_list and converted_list[0] and converted_list[0]["gender"]:
        return converted_list[0]["gender"]
    else:
        return None


def gender_of_second_cast(all_cast):
    cleaned = all_cast.replace("/", "")
    converted_list = list(eval(cleaned))
    if converted_list and len(converted_list)>1 and converted_list[1] and converted_list[1]["gender"]:
        return converted_list[1]["gender"]
    else:
        return None


def production_companies(all_production):
    cleaned = all_production.replace("/", "")
    list_of_companies = list(eval(cleaned))
    s = ""
    for company in list_of_companies:
        if len(s) > 0:
            s += "|"
        s += company["name"]
    return s


def release_year(date):
    if date:
        return int(date[:4])
    return date


def release_month(date):
    if date:
        return int(date[5:7])
    return date


def imdb_title(title):
    return title.split("\xa0")[0]


# IMDB database
imdb_dataset = load_df_from_csv("datasets/imdb_movie_metadata.csv")
imdb_dataset = imdb_dataset.drop("color", "director_name", "director_facebook_likes", "num_critic_for_reviews",
                                 "actor_3_facebook_likes", "actor_2_name", "actor_1_name", "num_voted_users",
                                 "actor_3_name", "facenumber_in_poster", "plot_keywords", "movie_imdb_link",
                                 "num_user_for_reviews", "language", "country", "title_year", "actor_2_facebook_likes",
                                 "aspect_ratio", "actor_1_facebook_likes", "gross")

imdb_dataset = imdb_dataset.withColumnRenamed("movie_title", "imdb_movie_title")

udf_imdb_title = udf(imdb_title, StringType())
imdb_dataset = imdb_dataset.withColumn("imdb_movie_title", udf_imdb_title("imdb_movie_title"))

# TMDB credits database
crew_dataset = load_df_from_csv("datasets/" + "tmdb_5000_credits.csv")

udf_cast_num = udf(json_list_num, IntegerType())
udf_first_cast_gender = udf(gender_of_first_cast, IntegerType())
udf_cast_num = udf(json_list_num, IntegerType())
udf_first_cast_gender = udf(gender_of_first_cast, IntegerType())
udf_second_cast_gender = udf(gender_of_second_cast, IntegerType())

crew_dataset = crew_dataset.withColumn("cast_number", udf_cast_num("cast")) \
    .withColumn("cast_number", udf_cast_num("crew")) \
    .withColumn("first_cast_gender", udf_first_cast_gender("cast")) \
    .withColumn("second_cast_gender", udf_second_cast_gender("cast"))

crew_dataset = crew_dataset.drop("cast", "crew")
crew_dataset = crew_dataset.withColumnRenamed("title", "tmdb_movie_title")

# TMDB Movie dataset
tmdb_dataset = load_df_from_csv("datasets/" + "tmdb_5000_movies.csv")

tmdb_dataset = tmdb_dataset.select("production_companies", "title", "release_date", "vote_average", "revenue", "id")

udf_production_companies = udf(production_companies, StringType())
tmdb_dataset = tmdb_dataset.withColumn("production_companies", udf_production_companies("production_companies"))

udf_release_year = udf(release_year, IntegerType())
udf_release_month = udf(release_month, IntegerType())
tmdb_dataset = tmdb_dataset.withColumn("release_year", udf_release_year("release_date"))
tmdb_dataset = tmdb_dataset.withColumn("release_month", udf_release_month("release_date"))

tmdb_dataset = tmdb_dataset.drop("release_date")

# merge the 3 datasets

two_tmdb_joined = tmdb_dataset.join(crew_dataset, crew_dataset.movie_id == tmdb_dataset.id, "inner").drop("id").drop(
    "movie_id")
dataset = two_tmdb_joined.join(imdb_dataset, two_tmdb_joined.title == imdb_dataset.imdb_movie_title, "inner").drop(
    "imdb_movie_title").drop("tmdb_movie_title")

# remove all null values
cols = dataset.columns
for col in cols:
    dataset = dataset.filter(dataset[str(col)].isNotNull())

#print the final schema
print(dataset.count()) # we end up with 3811 movies
print(dataset.printSchema())
print()

# 
# Noah: get genre
#
def toCSVLineRDD(rdd):
    '''
    This function convert an RDD or a DataFrame into a CSV string
    '''
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row]))\
           .reduce(lambda x,y: os.linesep.join([x,y]))
    return a + os.linesep
def toCSVLine(data):
    '''
    Convert an RDD or a DataFrame into a CSV string
    '''
    if isinstance(data, RDD):
        return toCSVLineRDD(data)
    elif isinstance(data, DataFrame):
        return toCSVLineRDD(data.rdd)
    return None

print('Trying to print "dataset":')

# dataset.show()
# dataset.toPandas().to_csv('output-dataset.csv') 

# TEST Add new columns, one for each genre, for each movie
dataset_with_newcolumn = dataset.select(split(dataset.genres, '[|]').alias('genres_arr')).collect()
print(dataset_with_newcolumn[20])

# TEST Get all the movie genre and store them in some list 
getAllGenres = dataset.select('title', 'genres')
getAllGenres.show()
getAllGenres_rdd = getAllGenres.rdd
getAllGenres_rdd = getAllGenres_rdd.map(lambda x: (x[0], x[1].split("|")))

fixedListOfGenres = getAllGenres_rdd.flatMap(lambda x: (x[1])).distinct()
print(fixedListOfGenres.collect()) # !! This list all the distinct genres from our dataset!


# fixedListOfGenres_list = fixedListOfGenres.groupByKey().collect()[1:10]
# print(fixedListOfGenres_list)
first10 = getAllGenres_rdd.collect()[1:10]
# print(first10)
getAllGenres_df = getAllGenres_rdd.toDF(["Title", "Genres array"])
getAllGenres_df.printSchema()
getAllGenres_df.show(10)

# TEST 2: Convert dataset into new dataset with 12 ish new column for the 11 fixed genres + misc OR get from grouping the dataset
fixed_genres = ["action", "adventure" , "drama", "horror", "comedy", "romantic", "historical", "thriller", "mystery", "sci-fi", "fantasy", "family", "sport", "Other"]


# TODO: write a function that maps different genres variation to the fixed genres

# getAllGenres.toPandas().to_csv('output_genres.csv')


# Processing gender

# test = dataset.select('title', 'content_rating').where(dataset.content_rating != 'PG')
# test.show()
# test.toPandas().to_csv('get_weird_content_rating.csv')


