<p align="center"><h2>IMPLEMENETATION OF A MACHINE LEARNING MODEL FOR OPTIMAL RETAIL BUSINESS PRICE BRANDING IN A COMPETITIVE ENVIRONMENT</h2></p>

## Project Overview
The goal of this project is to build a machine learning model to optimize retail pricing in a competitive business environment. By leveraging historical sales data and features related to products, competitors, and market conditions, the project aim to predict the optimal pricing strategy that maximizes sales and revenue for the company.
## Buisness Objective
* Maximize Revenue: Optimize pricing strategies to increase overall revenue.
*	Competitive Analysis: Understand how competitor pricing affects demand and adjust our prices accordingly.
*	Demand Forecasting: Accurately predict sales quantities based on price changes and other influencing factors.
*	Seasonal Trends: Adjust pricing based on seasonal variations, holidays, and weekends.
## Data Collection
The data “Retail Sales and Price Optimization” used for this project was sourced from Kaggle, a popular platform for datasets and data science competitions. Kaggle provides a wide range of publicly available datasets across various domains, including retail, finance, healthcare, and more. Kaggle datasets are often cleaned and well-documented, making them a reliable source for building and testing machine learning models. The dataset used in this project is publicly available and does not contain personally identifiable information (PII), ensuring compliance with data privacy standards. The dataset was checked for any usage restrictions. According to the Kaggle dataset license, it is freely available for educational and research purposes.
## Description of Dataset
The dataset consists of 676 rows and 30 columns representing various features related to retail pricing, demand, and market conditions. It contains detailed information on product pricing, demand, competitor pricing, freight costs, and other features spanning from January 2017 to August 2018. It includes both numerical and categorical variables, making it suitable for a mix of regression and classification analyses. Below is an explanation of each feature and its importance in the context of price optimization:
* **customers (integer)**: Represents the monthly demand for a specific subcategory of goods, measured by the number of customers. It is a key indicator of the relationship between pricing and demand, helping to gauge the price sensitivity of different products.
* **freight_price (float)**: The company’s freight cost for delivering the goods. Freight prices can influence the total cost for the company and thus impact pricing decisions. Higher freight costs may require higher prices to maintain profitability.
* **fp1, fp2, fp3 (float)**: Freight prices for competitors 1, 2, and 3, respectively. These provide insight into how competitors are managing their logistics costs, which can affect their pricing strategies and competitiveness in the market.
* **product_category_name (categorical)**: The broad category to which the product belongs (e.g., electronics, clothing). Understanding the category helps in modeling different demand patterns across product types, as pricing sensitivity can vary widely across categories.
* **product_id (categorical)**: A unique identifier for each detailed subcategory of goods. This feature allows for granular analysis of pricing trends and demand at the subcategory level.
* **product_description_lenght (integer)**: The number of words in the product subcategory description. A detailed product description can influence customer purchase decisions by providing more information, which might affect the perceived value of the product.
* **product_score (float)**: The user rating for the company's goods in the subcategory. Ratings can significantly impact demand, as higher ratings generally indicate better quality or customer satisfaction, leading to higher demand or allowing for premium pricing.
* **ps1, ps2, ps3 (float)**: User ratings for the goods of competitors 1, 2, and 3, respectively. These ratings allow the model to account for how customer perceptions of competitors' products influence the demand for the company's goods.
* **product_photos_qty (integer)**: The number of photos provided for each subcategory. More product images can enhance customer confidence and increase demand, especially in online retail environments.
* **product_weight_g (integer)**: The weight of each unit in grams. This affects shipping costs and can influence customer demand, particularly in cases where shipping fees are passed on to the consumer.
* **total_price (float)**: Monthly revenue, calculated using the formula: total_price = unit_price * qty. It represents the outcome the model aims to optimize, as the goal is to maximize total revenue by finding the optimal price point.
* **month_year (string)**: The date information in the format (dd-mm-yyyy), which is later split into year and month. This feature helps in identifying seasonal trends and understanding how sales vary over time.
* **year (integer) & month (integer)**: Extracted from month_year, these variables represent the temporal aspect of the data. They help capture seasonality and the long-term evolution of prices and demand.
* **qty (integer)**: The quantity of goods sold per month per subcategory. This is an essential feature for understanding the relationship between price and demand, as well as for calculating revenue.
* **unit_price (float)**: The price at which the company sells a unit of a subcategory's goods. This is the key feature to be optimized in the model to maximize total revenue or profit.
* **comp_1, comp_2, comp_3 (float)**: Unit prices for the same subcategory of goods from competitors 1, 2, and 3. These features allow the model to factor in competitive pricing strategies and understand how the company's pricing compares in the market.
* **lag_price (float)**: The unit price of the product subcategory from the previous month. This feature helps in capturing the temporal effects of price changes, as past prices can influence current demand through customer expectations and loyalty.
* **weekend (integer), weekday (integer), holiday (integer)**: These features represent the number of weekends, weekdays, and holidays in a given month. These temporal features are useful for modeling fluctuations in demand, as consumer purchasing behavior can change based on the time of the week or during holidays.
## Data Cleaning
* Missing values: Checked for missing values to ensure data quality and the result shows that there were no missing values.
* Duplicate values: Checked for duplicate values to ensure data quality and the result shows that there were no duplicate values.
## Data Handling
* Converted month_year to year and month (already provided) and ensure all time-based features are in an appropriate format.
* Checked for outliers to ensure data quality and result show that there were outliers, but due to their large numbers they were not removed but handled by centering and scaling of each features. 
## Data Normalization and Transformation
* Centering: Centered numerical features so that the mean of the features become zero this helps to removed bias that can result from features with large mean value.
* Scaling: Scaled numerical features so that their range lie within a specific range this helps to avoid biased model performance which result from features with larger ranges dominating over features with smaller ranges.  
* Standardization: The data was standardize using StandardScaler to ensure all features are on the same scale.
## Feature Engineering
New feature Created
* **Weekend ratio**:created to indicate how “weekend-heavy” a month is.
* **Price difference with competitors**: created to indicate the price difference between the company’s products and each competitor. This helps the model capture how changes in competitive pricing affect demand.
* **Competitive price index**: this is an aggregate feature created to combines all competitor price differences into one value.
* **Freight ratio**: this is an aggregate feature created to compares the company’s freight price to the average freight price of competitors.
* **Ratings ratio**: this is an aggregate feature created to compares the company’s user's ratings to the average user's ratings of competitors.
* **Product weight to competitor’s price**: created to indicate how the company’s product weight varied with competitor’s price.
* **Competitive product weight index**: this is an aggregate feature that indicate the average product weight to competitors’ price ratio.
* **Product name length to competitors’ price**: created to indicate how the company’s product name length varied with competitor’s price.
* **Competitive product name length index**: this is an aggregate feature that indicate the average product name length to competitors’ price ratio.
* **Product description length to competitors’ price**: created to indicate how the company’s product description length varied with competitor’s price.
* **Competitive product description length index**: this is an aggregate feature that indicate the average product description length to competitors’ price ratio.
* **Holiday month**: this feature indicate a binary flag for month with important holidays (e.g. December for Christmas, November for black Friday).
## Model Selection
The machine learning models applied are: **Decision Tree** and **Random Forest**.
### **Decision Trees** 
A supervised machine learning algorithm used for classification and regression tasks. It mimics human decision-making by splitting data into subsets based on feature values, forming a tree-like structure. Decision tree is a tree-based algorithm capable of identifying nonlinear relationships between variables and making predictions through a series of if-then rules. It can also be utilized to segment customers based on their behavior and optimize pricing strategies effectively.
### Why Decision Trees
* Actionable Insights: Generate interpretable rules for pricing decisions.
* Data-Driven Pricing: Adjust prices dynamically based on real-time data.
* Customer Behavior Understanding: Segment customers for targeted promotions.
* Profit Maximization: Identify optimal price points balancing demand and profit.
* Scalability: Applicable across multiple product categories and markets.
### **Random Forest**
An ensemble learning method used for both classification and regression tasks. It builds multiple decision trees during training and combines their outputs (via averaging for regression or majority voting for classification) to improve accuracy and reduce overfitting. It is a robust and versatile algorithm that works well with both categorical and numerical data.
### Why Random Forest
* Handles High-Dimensional Data: Efficiently works with large datasets containing numerous features.
* Robust to Noise: Effective with imperfect or missing data.
* Nonlinear Insights: Captures complex relationships between price and other factors.
* Scalable: Applicable across diverse product categories and large-scale retail datasets.
* Feature Importance: Provides insights into which variables most influence pricing decisions.
## Model Building
### Creation of Sub-data
Nine Sub-data were created from the data after feature selection. The sub-data were created based on different product categories (garden_tools, health_beauty, watches_gifts, computers_accessories, bed_bath_table, cool_stuff, furniture_decor, perfumery, and consoles_games). Models were built to predict the unit price for each sub-category. 
### Data Splitting
The dataset was split into training and testing sets using an 80/20 split. The models (Random Forest, and Decision Tree) were initiated.
### Hyperparameter Grid
Hyperparameter grid was set up for the Decision Tree model to explore different values for tuning. GridSearchCV was used to perform an exhaustive search over the hyperparameter space. The model is fitted with the best hyperparameter found.
Hyperparameter grid was not set up for the Random Forest model due to its high processing time but hyperparameter such as n_jobs and random_state was used to improve the model performance.
## Model Evaluation
The model is evaluated using two performance metrics –mean absolute error (MAE) and R² score for each model on the test set.
* Mean Absolute Error (MAE): MAE is the average of the absolute differences between predicted and actual values. It measures the magnitude of errors in a set of predictions, without considering their direction (positive or negative).
* R² Score (Coefficient of Determination): R² measures the proportion of variance in the dependent variable that is predictable from the independent variables. It indicates how well the model explains the variability of the target variable.
## Model Deployment
Random Forest model was used for the deployment because it provide the most accurate predictions with the highest R² score and the lowest error metrics compared to Decision Tree model as seen in the tables above.
### Streamlit Library
The optimized model was deployed using Streamlit for price prediction and demand forecasting. Streamlit is a popular Python library used for creating web applications with minimal effort, particularly in the context of data science and machine learning. It's use for deployment of this project because of the following reasons:
* Ease of Use: Streamlit is designed to be user-friendly and requires minimal code to create interactive web applications. This is particularly beneficial for data scientists and machine learning practitioners who may not have extensive web development experience.
* Integration with Data Science Libraries: Streamlit seamlessly integrates with popular data science libraries and frameworks such as Pandas, Matplotlib, Plotly, and scikit-learn. This makes it easy to incorporate data visualizations and machine learning models into the web application.
* Built-in Widgets: Streamlit provides a variety of built-in widgets, such as sliders, buttons, and text inputs, making it easy to create interactive elements within the application. Users can interact with the application to explore different input parameters for the machine learning models.
* Customization: while streamlit is easy to use, it also allows for customization. The appearance and layout were customize by incorporating HTML and CSS for advanced styling.
## Future Improvement
* Incorporate Real-Time Data: The dataset covers historical data up to August 2018, which may not reflect current market trends or pricing strategies. Integrate real-time data streams to keep the model updated with the latest market conditions, competitor pricing, and seasonal trends.
* Dynamic Pricing Using Reinforcement Learning: The model relies on static historical data and provides predictions based on a predefined dataset. Implement reinforcement learning techniques for dynamic pricing, where the model learns optimal pricing strategies over time by interacting with the environment (e.g., adjusting prices and observing sales).
* Integration of External Factors: The dataset lacks external influencing factors like economic indicators, weather conditions, or social media trends. Enrich the dataset with external data to capture broader market dynamics.
* Explore Deep Learning Models: The project uses traditional machine learning models (Random Forest, Decision Tree, etc.). Experiment with deep learning models, such as Neural Networks or Recurrent Neural Networks (RNNs), for more complex, non-linear relationships.
