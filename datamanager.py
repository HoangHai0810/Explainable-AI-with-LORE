import numpy as np
import pandas as pd

from collections import defaultdict


def prepare_dataset(df, class_name):

    df = remove_missing_values(df)

    numeric_columns = get_numeric_columns(df)

    rdf = df

    df, feature_names, class_values = one_hot_encoding(df, class_name)

    real_feature_names = get_real_feature_names(rdf, numeric_columns, class_name)

    rdf = rdf[real_feature_names + (class_values if isinstance(class_name, list) else [class_name])]

    features_map = get_features_map(feature_names, real_feature_names)

    return df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map


def get_features_map(feature_names, real_feature_names):
    features_map = defaultdict(dict)
    i = 0
    j = 0
    while i < len(feature_names) and j < len(real_feature_names):
        #Nếu tên 2 cột thuộc tính giống nhau: 
        if feature_names[i] == real_feature_names[j]:
            # Loại bỏ phần tiền tố được thêm vào khi thực hiện one-hot encoding sau đó lưu vào bản đồ
            # keys = j, values = {'Thuộc tính': i }
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
            j += 1
        #Nếu tên cột thuộc tính trong biến feature_names (sau khi OHE) bắt đầu bằng từ của cột thuộc tính ban đầu
        elif feature_names[i].startswith(real_feature_names[j]):
            # Loại bỏ phần tiền tố được thêm vào khi thực hiện one-hot encoding sau đó lưu vào bản đồ
            # keys = j, values = {'Thuộc tính': i }
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
        else:
            j += 1
    return features_map


def get_real_feature_names(rdf, numeric_columns, class_name):
    if isinstance(class_name, list):
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c not in class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c not in class_name]
    else:
        #Thêm toàn bộ tên cột kiểu số và kiểu phân loại vào list rồi lưu vào real_features_names.
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c != class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c != class_name]
    return real_feature_names


def one_hot_encoding(df, class_name):
    if not isinstance(class_name, list):
        # dfX thực hiện One_hot_encoding (ứng với những sample có giá trị giống với giá trị của cột thì vị trí đó được đánh số 1)
        dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep='=')
        # Lấy ra các giá trị khác nhau của biến target và gán số cho từng giá trị đó: (High Price = 0, Low Price = 1)
        class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
        # Thay thế các giá trị đó bằng giá trị số tương ứng
        dfY = df[class_name].map(class_name_map)
        # Nhập dfX và dfY lại tạo thành dataframe hoàn chỉnh sau khi One_hot_encoding.
        df = pd.concat([dfX, dfY], axis=1).reindex(dfX.index)
        feature_names = list(dfX.columns) # Lưu toàn bộ biến độc lập vào kiểu danh sách sau đó lưu vào biến feature_name
        class_values = sorted(class_name_map) # Sắp xếp lại giá trị số trong biến class_name_map
    else: # isinstance(class_name, list)
        dfX = pd.get_dummies(df[[c for c in df.columns if c not in class_name]], prefix_sep='=')
        # class_name_map = {v: k for k, v in enumerate(sorted(class_name))}
        class_values = sorted(class_name)
        dfY = df[class_values]
        df = pd.concat([dfX, dfY], axis=1).reindex(dfX.index)
        feature_names = list(dfX.columns)
    return df, feature_names, class_values


def remove_missing_values(df):
    for column_name, nbr_missing in df.isna().sum().to_dict().items():
        if nbr_missing > 0:
            if column_name in df._get_numeric_data().columns:
                mean = df[column_name].mean()
                df[column_name].fillna(mean, inplace=True)
            else:
                mode = df[column_name].mode().values[0]
                df[column_name].fillna(mode, inplace=True)
    return df


def get_numeric_columns(df):
    numeric_columns = list(df._get_numeric_data().columns)
    return numeric_columns


def prepare_iris_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, skipinitialspace=True)
    return df, class_name


def prepare_wine_dataset(filename):
    class_name = 'quality'
    df = pd.read_csv(filename, skipinitialspace=True, sep=';')
    return df, class_name


def prepare_adult_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
    columns2remove = ['fnlwgt', 'education-num']
    df.drop(columns2remove, inplace=True, axis=1)
    return df, class_name


def prepare_german_dataset(filename):
    class_name = 'default'
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = [c.replace('=', '') for c in df.columns]
    return df, class_name


def prepare_compass_dataset(filename, binary=False):

    df = pd.read_csv(filename, delimiter=',', skipinitialspace=True)

    columns = ['age', 'age_cat', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    if binary:
        def get_class(x):
            if x < 7:
                return 'Medium-Low'
            else:
                return 'High'
        df['class'] = df['decile_score'].apply(get_class)
    else:
        df['class'] = df['score_text']

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    class_name = 'class'
    return df, class_name


def prepare_churn_dataset(filename):
    class_name = 'churn'
    df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
    columns2remove = ['phone number']
    df.drop(columns2remove, inplace=True, axis=1)
    return df, class_name

def prepare_ANTT_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, skipinitialspace = True, keep_default_na = True)
    df = df.drop(df.columns[0],axis = 1)
    new_col = []
    for item in df['Price (£)']:
        if 0 < item <= 30000:
            new_col.append('Low Price')
        else:
            new_col.append('High Price')
    df[class_name] = new_col
    return df, class_name
