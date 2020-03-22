# Name: Chenxi Zhu



import pandas as pd
import logging

raw_data_path = "./data/nyc-rolling-sales.csv"
clean_data_path = "after_processing_rolling_sales.csv"
log_file = "./logs/data_processing_info.log"

# boroughs mapping
id_borough_mapping = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn",
					  4: "Queens", 5: "Staten Island"}


def main():
	# print ("hello world")
	logging.basicConfig(
    filename=log_file,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)

	raw_data = pd.read_csv(raw_data_path)
	raw_data.columns = map(str.lower, raw_data.columns)
	raw_data.columns = map(lambda s: s.replace(" ", "_"), raw_data.columns)
	cols_to_drop = ["unnamed:_0", "ease-ment"]
	raw_data.drop(cols_to_drop, axis="columns", inplace=True)
	raw_data["borough_name"] = raw_data["borough"].map(id_borough_mapping)

	for column_name in ['sale_price', 'land_square_feet', 'gross_square_feet']:
		raw_data[column_name] = pd.to_numeric(raw_data[column_name], errors='coerce')


	raw_data["sale_date"] = pd.to_datetime(raw_data["sale_date"])
	raw_data["sale_year"] = raw_data["sale_date"].dt.year
	raw_data["sale_month"] = raw_data["sale_date"].dt.month
	raw_data["sale_age"] = raw_data["sale_year"] - raw_data["year_built"]

	categorical_list = ["tax_class_at_time_of_sale", "tax_class_at_present",
                 		"zip_code", "sale_year", "sale_month"]
	for column_name in categorical_list:
		raw_data[column_name] = raw_data[column_name].astype("category")

	msg = "There are {:.0f} duplicated rows which will be dropped.".format(
        sum(raw_data.duplicated(raw_data.columns))
        )
	logging.info(msg) 

	raw_data = raw_data.drop_duplicates(raw_data.columns)

	msg = "{:.0f} rows without selling price info were dropped.".format(
        sum(raw_data["sale_price"].isnull())
        )
	raw_data = raw_data[raw_data["sale_price"].notnull()]
	logging.info(msg)

	msg = "{:.0f} rows with a selling price of $10 or below were dropped.".format(
        sum(raw_data["sale_price"] <= 10)
        )
	raw_data = raw_data[raw_data["sale_price"] > 10]
	logging.info(msg)

	msg = "{:.0f} rows with a construction date before 1900 were dropped.".format(
        sum(raw_data["year_built"] < 1800)
        )
	raw_data = raw_data[raw_data["year_built"] > 1800]
	logging.info(msg)

	msg = "{:.0f} rows with a size of zero square feet were dropped.".format(
        sum(~(raw_data["land_square_feet"] > 0))
        )
	raw_data = raw_data[raw_data["land_square_feet"] > 0]
	logging.info(msg)
	first_lists = ["residential_units", "commercial_units", "land_square_feet", "sale_price", "sale_age"]
	second_lists = ["neighborhood", "building_class_category", "tax_class_at_time_of_sale", "borough_name", "sale_year", "sale_month"]
	clean_lists = first_lists + second_lists
	one_hot_encoded = pd.get_dummies(raw_data[second_lists])
	raw_data_temp = raw_data[clean_lists]
	raw_data_temp.drop(second_lists, axis=1, inplace=True)
	raw_data_clean = pd.concat([raw_data_temp, one_hot_encoded], axis=1)
	del raw_data_temp

	raw_data_clean.sort_values(by=["sale_price"], inplace=True)
	raw_data_clean.reset_index(drop=True, inplace=True)
	raw_data_clean = raw_data_clean.astype("float64")

	raw_data_clean.apply(lambda x: logging.info(x.describe()), axis=0)
	raw_data_clean.to_csv("./data/" + clean_data_path, index=False)





if __name__ == "__main__":
	main()