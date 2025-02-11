import numpy as np
from sklearn.impute import SimpleImputer


# class for cleaning dataset
class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="most_frequent", missing_values=np.nan)

    def clean_data(self, data):
        # drop unnecessary columns
        data.drop(['id', 'SalesChannelID', 'VehicleAge', 'DaysSinceCreated'], axis=1, inplace=True)

        data["AnnualPremium"] = data['AnnualPremium'].str.replace('£', '').str.replace(',', '').astype('float')

        for col in ['Gender', 'RegionID']:
            data[col] = self.imputer.fit_transform(data[[col]]).flatten()
        # fill missing values
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['HasDrivingLicense'] = data['HasDrivingLicense'].fillna(1)
        data['Switch'] = data['Switch'].fillna(-1)
        data['PastAccident'] = data['PastAccident'].fillna("Unknown", inplace=False)

        # remove outliers
        Q1 = data['AnnualPremium'].quantile(0.25)
        Q3 = data['AnnualPremium'].quantile(0.75)
        IOR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IOR
        data = data[data['AnnualPremium'] <= upper_bound]
        return data
