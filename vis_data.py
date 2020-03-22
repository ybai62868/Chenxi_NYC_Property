# Chenxi Zhu

import data_processing
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

clean_data_path = './data/after_processing_rolling_sales.csv'

def main():
	clean_data = pd.read_csv(clean_data_path)
	sale_price_pct95 = np.percentile(clean_data["sale_price"], 95)
	plt.style.use("seaborn")

	fig, ax = plt.subplots(1, 2)
	ax[0].hist(clean_data[clean_data["sale_price"] < sale_price_pct95]["sale_price"], bins=100)
	ax[0].set(title = "The data is below 95% of all prices",
			  xlabel = "Sale price in $",
			  ylabel = "Frequency")
	ax[1].hist(clean_data["sale_price"].apply(np.log), bins=100)
	ax[1].set(title = "Log transformation sale price",
			  xlabel = "Log(selling price) in $",
			  ylabel = "Frequency")

	fig.suptitle("Selling prices are approximately log-normally distributed", size=16)

	fig.subplots_adjust(top=0.85)
	fig.savefig("./figs/sale_price_dis_log.png", dpi=1000)
	plt.close(fig)






	# data_processing.id_borough_mapping


if __name__ == "__main__":
	main()