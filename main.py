import csv
import os
import sys
# Spark imports
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, udf

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
imdb_dataset = load_df_from_csv("datasets\imdb_movie_metadata.csv")
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

# Possible other method to categorize. This one unfortunately only puts 1's and 0's but doesn't create a new column.
# Might need to update loop

# distinct_rating = dataset.select('content_rating').distinct().rdd.flatMap(lambda x: x).collect()
#
# for distinct_rating in distinct_rating:
#     function = udf(lambda item: 1 if item == distinct_rating else 0, IntegerType())
#     new_column_rating = "content_rating"+'_'+distinct_rating
#     new_dataset = dataset.withColumn(new_column_rating, function("content_rating"))
#
# new_dataset.show()


indexer = StringIndexer(inputCol='content_rating', outputCol='ContentIndex')
indexed = indexer.fit(dataset).transform(dataset)
indexed.show(100, truncate=False)
