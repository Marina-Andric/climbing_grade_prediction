import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from routes_sql_queries import sql_user_features, sql_route_features, sql_mad_function, sql_mad_function_1year, sql_features_routes

np.random.seed(42)

GRADE_OFFSET = 1
SHRINKAGE = 4

# Global variables
# conn = psycopg2.connect(host='localhost', port=5432, user='postgres', password='postgres',
#                         dbname='vertikal_life_18122020')

conn = psycopg2.connect(host='localhost', port=5432, user='andric', password='Ufit3quac/',
                        dbname='andric')
cur_pg = conn.cursor()


def get_data_climber(table_name):

    sql = '''
    select * from {}
    order by date asc;
    '''
    cur_pg.execute(sql.format(table_name))
    df = pd.DataFrame(cur_pg.fetchall())
    df.columns = [item[0] for item in cur_pg.description]
    # df.to_csv("Data/gym_routes_climber_grade_prediction.csv")
    return df


def compute_user_error(y_test, y_pred_round, X_test, user_ids, table, t):
    # print(table, t)
    users_error_regression = {}
    users_error_baseline = {}
    print('included ', len(np.unique(user_ids)))

    for i, user_id in enumerate(user_ids):
        if user_id not in users_error_regression.keys():
            users_error_regression[user_id] = {'predictions': [], 'truth': [], 'errors': [], 'count': 0}
            users_error_regression[user_id]['errors'].append(y_test[i] - y_pred_round[i])
            users_error_regression[user_id]['predictions'].append(y_pred_round[i])
            users_error_regression[user_id]['truth'].append(y_test[i])
            users_error_regression[user_id]['count'] += 1

            users_error_baseline[user_id] = {'predictions': [], 'truth': [], 'errors': [], 'count': 0}
            users_error_baseline[user_id]['errors'].append(y_test[i] - list(X_test['grade_id'])[i])
            users_error_baseline[user_id]['predictions'].append(list(X_test['grade_id'])[i])
            users_error_baseline[user_id]['truth'].append(y_test[i])
            users_error_baseline[user_id]['count'] += 1
        else:
            users_error_regression[user_id]['errors'].append(y_test[i] - y_pred_round[i])
            users_error_regression[user_id]['predictions'].append(y_pred_round[i])
            users_error_regression[user_id]['truth'].append(y_test[i])
            users_error_regression[user_id]['count'] += 1

            users_error_baseline[user_id]['errors'].append(y_test[i] - list(X_test['grade_id'])[i])
            users_error_baseline[user_id]['predictions'].append(list(X_test['grade_id'])[i])
            users_error_baseline[user_id]['truth'].append(y_test[i])
            users_error_baseline[user_id]['count'] += 1

    print("Number of users (regression): ", len(users_error_regression.keys()))
    print("Number of users (baseline): ", len(users_error_baseline.keys()))
    errors_reg = []
    errors_bas = []
    for user_id in users_error_regression.keys():
        errors_reg.append(round(mean_squared_error(users_error_regression[user_id]['predictions'],
                                                   users_error_regression[user_id]['truth'], squared=False), 2))
        errors_bas.append(round(mean_squared_error(users_error_baseline[user_id]['predictions'],
                                                   users_error_baseline[user_id]['truth'], squared=False), 2))

    print("avg and std of RMSE across users for reg.mod. ", np.round(np.mean(errors_reg),3), np.round(np.std(errors_reg),3))
    print("avg and std of RMSE across users for baseline ", np.round(np.mean(errors_bas),3), np.round(np.std(errors_bas),3))
    return users_error_regression, users_error_baseline


def predict_user_grade(X_test, model, scaler, features_user_selected, features_route_selected, features_setter, table, out_file_name, features_table, grade_range, shrinkage):
    # features_route_selected = [item for item in features if item in features_route]
    # features_user_selected = [item for item in features if item in features_user]
    features = features_user_selected + features_route_selected + features_setter
    test_data = pd.DataFrame(columns=features)
    test_data_complete = pd.DataFrame(columns=features)

    for i, row in X_test.iterrows():
        cur_pg.execute(sql_route_features.format(row['route_id'], row['date'], row['grade_id'], row['user_id'], table, shrinkage))
        df_route_features = pd.DataFrame(cur_pg.fetchall())
        if df_route_features.shape[0] == 0:
            print("No route features")
            df_route_features = pd.DataFrame(data = np.array([[row['grade_id'], 0, 0, 0, 0, 0]]), columns=features_route)
        else:
            df_route_features.columns = [item[0] for item in cur_pg.description]
            df_route_features = df_route_features[features_route]

        cur_pg.execute(sql_user_features.format(row['user_id'], row['date'], row['grade_id'], row['route_id'], table, grade_range, shrinkage))
        df_user_features = pd.DataFrame(cur_pg.fetchall())
        if df_user_features.shape[0] == 0:
            # print(df_user_features)
            print("No user features")
            df_user_features = pd.DataFrame(data = np.array([[0, 0, 0, 0, 0]]), columns=features_user)
        else:
            df_user_features.columns = [item[0] for item in cur_pg.description]
            df_user_features = df_user_features[features_user]

        # cur_pg.execute(sql_setter_features.format(row['route_setter_id'], row['date'], row['grade_id'], table_name, grade_range, shrinkage))
        # df_setter_features = pd.DataFrame(cur_pg.fetchall())
        # if df_setter_features.shape[0] == 0:
        #     df_setter_features = pd.DataFrame(data = np.array([[0, 0, 0]]), columns=features_setter)
        # else:
        #     df_setter_features.columns = [item[0] for item in cur_pg.description]
        #     df_setter_features = df_setter_features[features_setter]

        df_concat = pd.concat([df_route_features, df_user_features], axis=1)
        test_data = pd.concat([test_data, df_concat], axis=0)

    # save feature data in csv files
    test_data.to_csv("Data/test_data_" + str(out_file_name) + ".csv")
    X_test.to_csv("Data/test_ds_" + str(out_file_name) + ".csv")
    print("test_data, X_test shapes: ", test_data.shape[0], X_test.shape[0])
    # test_data_complete.to_csv("Data/test_data_complete_" + str(threshold) + ".csv")

    # test_data = pd.read_csv("Data/test_data" + str(threshold) + ".csv")

    test_data = test_data[features]

    scaled_features = scaler.transform(test_data.values)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features)

    y_pred = model.predict(scaled_features_df)
    y_pred_round = [round(item) for item in y_pred]

    return y_pred_round, X_test


def train_and_test(model, X, features_user, features_route, features_setter, table_name,
                   intermediate_table, query, out_file_name, features_table,
                   features_route_selected, features_user_selected, tr, grade_range, shrinkage):
    scaler = StandardScaler()
    features = features_user + features_route + features_setter
    # print("concatenated features: ", features)

    sql = '''insert into {}
            (user_id, train_route_id) values ({}, '{}');'''
    sql_empty_table = '''drop table if exists {0};
                         create table {0} (like aux_schema.gym_routes_intermediate including all);
                         --  delete from {0};
                        '''
    sql_update_test_routes = '''
        update {}
        set test_route_id = '{}' where user_id = {};
    '''
    cur_pg.execute(sql_empty_table.format(intermediate_table))

    for user_id in X['user_id'].unique():
        # break
        # print(user_id)
        user_routes = X[X['user_id'] == user_id]['route_id']

        if len(user_routes) >= tr:
            train = [route for i, route in enumerate(user_routes) if i < int(len(user_routes)*.8)]
            test = [route for route in user_routes if route not in train]

            db_lst_routes = '{'
            for route_id in train:
                db_lst_routes += str(route_id) + ','
            n = db_lst_routes.rfind(',')
            db_lst_routes = db_lst_routes[0:n] + db_lst_routes[n + 1:]
            db_lst_routes += '}'
            cur_pg.execute(sql.format(intermediate_table, user_id, db_lst_routes))

            db_lst_routes = '{'
            for route_id in test:
                db_lst_routes += str(route_id) + ','
            n = db_lst_routes.rfind(',')
            db_lst_routes = db_lst_routes[0:n] + db_lst_routes[n + 1:]
            db_lst_routes += '}'
            cur_pg.execute(sql_update_test_routes.format(intermediate_table, db_lst_routes, user_id))

    conn.commit()

    cur_pg.execute(query.format(table_name, intermediate_table, features_table))
    conn.commit()
    df = pd.DataFrame(cur_pg.fetchall())
    df.columns = [item[0] for item in cur_pg.description]

    train_ds= df[df['train'] == 1]
    # train_ds.dropna(axis=0, subset=features, inplace=True)
    # train_ds.to_csv("Data/complete_train_" + str(threshold) + ".csv")
    # exit()

    train_data = pd.DataFrame(columns=features)
    train_data_complete = pd.DataFrame(columns=features)

    for i, row in train_ds.iterrows():
        cur_pg.execute(sql_route_features.format(row['route_id'], row['date'], row['grade_id'], row['user_id'], table_name, shrinkage))
        df_route_features = pd.DataFrame(cur_pg.fetchall())
        if df_route_features.isnull().sum(axis=1)[0] > 0:
            print('No route features')
            df_route_features = pd.DataFrame(data = np.array([[row['grade_id'], 0, 0, 0, 0, 0, 0]]), columns=features_route)
        else:
            df_route_features.columns = [item[0] for item in cur_pg.description]
            df_route_features = df_route_features[features_route]

        cur_pg.execute(sql_user_features.format(row['user_id'], row['date'], row['grade_id'], row['route_id'], table_name, grade_range, shrinkage))
        df_user_features = pd.DataFrame(cur_pg.fetchall())
        if  df_user_features.isnull().sum(axis=1)[0] > 0:
            print("user id: ", row['user_id'])
            print(df_user_features)
            print("No user features")
            df_user_features = pd.DataFrame(data = np.array([[0, 0, 0, 0, 0]]), columns=features_user)
        else:
            df_user_features.columns = [item[0] for item in cur_pg.description]
            df_user_features = df_user_features[features_user]

        # cur_pg.execute(sql_setter_features.format(row['route_setter_id'], row['date'], row['grade_id'], table_name, grade_range, shrinkage))
        # df_setter_features = pd.DataFrame(cur_pg.fetchall())
        # if df_setter_features.isnull().sum(axis=1)[0] > 0:
        #     print("No route setter features")
        #     print(df_setter_features)
        #     exit()
        #     df_setter_features = pd.DataFrame(data = np.array([[0, 0, 0]]), columns=features_setter)
        # else:
        #     df_setter_features.columns = [item[0] for item in cur_pg.description]
        #     df_setter_features = df_setter_features[features_setter]
        # df_concat = pd.concat([df_route_features, df_user_features, df_setter_features], axis=1)

        df_concat = pd.concat([df_route_features, df_user_features], axis=1)
        train_data = pd.concat([train_data, df_concat], axis=0)

    train_data.to_csv("Data/train_data_" + str(out_file_name) + ".csv")
    train_ds.to_csv("Data/train_ds_" + str(out_file_name) + ".csv")
    print("train and train_ds shapes: ", train_data.shape, train_ds.shape)
    # train_data_complete.to_csv("Data/train_data_complete_" + str(threshold) + ".csv")

    # train_data = pd.read_csv("Data/train_data" + str(threshold) + ".csv")
    train_data = train_data[features]

    scaled_features = scaler.fit_transform(train_data.values)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features)

    X_train = scaled_features_df
    y_train = train_ds['user_grade_id']

    test_data = df[df['train'] == 0]
    print('test data shape: ', test_data.shape)

    # df_test = df[df['train'] == 0]['user_grade_id']
    X_test, y_test = test_data, df[df['train'] == 0]['user_grade_id']

    model.fit(X_train, y_train)
    # y_pred = model.predict(X_train)
    # y_pred_round = [round(item) for item in y_pred]

    y_pred_round, X_test = predict_user_grade(X_test, model, scaler, features_user, features_route, features_setter,
                                              table_name, out_file_name, features_table, grade_range, shrinkage)

    y_test = X_test['user_grade_id'].values
    user_ids_test = X_test['user_id'].values

    # print(y_test, y_pred_round, user_ids_test)
    users_error, users_error_baseline = compute_user_error(y_test, y_pred_round, X_test,
                                                           user_ids_test, table_name, tr)

    mae_regression = mean_absolute_error(y_test, y_pred_round)
    rmse_regression = mean_squared_error(y_test, y_pred_round, squared=False)

    mae_baseline = mean_absolute_error(y_test, list(X_test['grade_id']))
    rmse_baseline = mean_squared_error(y_test, list(X_test['grade_id']), squared=False)

    return rmse_regression, mae_regression, rmse_baseline, mae_baseline, \
           users_error, users_error_baseline


def performance_evaluation(users_error_baseline, users_error_regression):
    sample_size = len(users_error_regression.keys())
    sample_success = 0
    errors_baseline = []
    errors_regression = []
    users_success = {'users': []}
    users_failures = {'users': []}
    users_equal_error = {'users': []}

    for user_id in users_error_regression.keys():
        u_rmse_regression = mean_squared_error(users_error_regression[user_id]['predictions'],
                                               users_error_regression[user_id]['truth'], squared=False)
        u_rmse_baseline = mean_squared_error(users_error_baseline[user_id]['predictions'],
                                             users_error_baseline[user_id]['truth'], squared=False)
        errors_baseline.append(u_rmse_baseline)
        errors_regression.append(u_rmse_regression)
        if u_rmse_regression < u_rmse_baseline:  #
            sample_success += 1
            users_success['users'].append(user_id)
        elif u_rmse_regression == u_rmse_baseline:
            users_equal_error['users'].append(user_id)  # print(user_id, u_rmse_regression, u_rmse_baseline)
        else:
            users_failures['users'].append(user_id)

    print('better: ', len(users_success['users']))
    print('worse: ', len(users_failures['users']))
    print('equal: ', len(users_equal_error['users']))

    # Null hypothesis is that regression model generated more accurate prediction than baseline for 90 % of users
    # Alternative hypothesis is reg.model is worse than baseline for more than 90% of users
    null_hypothesis = 1 - round(sample_success / sample_size, 2)
    print("sample_success ", sample_success)
    print("sample_size ", sample_size)
    # stat, pval = proportions_ztest(sample_success, sample_size, 0.05)
    # stat, p_value = proportions_ztest(count=sample_success, nobs=sample_size, value=null_hypothesis, alternative='smaller')
    # print("Proportion ztest", round(stat, 3), round(p_value, 3))
    # w, p = wilcoxon(errors_regression, errors_baseline)
    # print(w, p)
    # w1, p1 = wilcoxon(errors_regression, errors_baseline, alternative='less')  # print("Wilcoxon test: ", round(w1,3), round(p1,3))


def regression_climber_grade(data, table_name, intermediate_table, query, features_user, features_route, features_setter, model,
                             threshold, features_table, features_route_selected, features_user_selected, tr, grade_range, shrinkage):
    features = features_user + features_route + features_setter
    print("features: ", features)
    print("Number of points: ", data.shape[0])

    X = data[['user_id', 'route_id', 'grade_id', 'user_grade_id']]

    results_regression_rmse, results_regression_mae, results_baseline_rmse, results_baseline_mae, \
    users_error_regression, users_error_baseline = train_and_test(model, X, features_user, features_route, features_setter,
                                                                  table_name, intermediate_table, query, threshold,
                                                                  features_table,
                                                                  features_route_selected, features_user_selected, tr, grade_range, shrinkage)
    performance_evaluation(users_error_baseline, users_error_regression)

    print("Baseline RMSE: ", round(results_baseline_rmse,3))
    print("Baseline MAE: ", round(results_baseline_mae,3))

    print("Regression RMSE: ", round(results_regression_rmse,3))
    print("Regression MAE: ", round(results_regression_mae,3))


threshold = 3
out_file_name = '_routes_v1'
table_name = 'aux_schema.routes_selected_climbs_v1'
# table_name = 'aux_schema.gym_routes_selected_climb_percentage'
intermediate_table = 'aux_schema.gym_routes_intermediate' + 'v1'
query = sql_features_routes
features_table = 'aux_schema.gym_routes_features' + 'v1'

model = linear_model.LinearRegression()
# model = RandomForestRegressor(n_estimators=4000, warm_start=True,
#                                                 max_depth=7, max_features=4,
#                                                 max_samples=0.3)

features_user = ['user_grade_mad', 'user_dev_sign',  'season', 'month', 'year']
features_route = ['grade_id', 'route_grade_mad', 'route_dev_sign',
                  'route_grade_mad_1year', 'route_dev_sign_1year', 'route_dev_sign_1year_season', 'altitude', 'length', 'rock_type'] # 'route_grade_mad',
# features_setter = ['rs_grade_mad', 'rs_dev_sign']
features_setter=[]

data = get_data_climber(table_name)

cur_pg.execute(sql_mad_function.format('route_id', 'route', table_name))
cur_pg.execute(sql_mad_function_1year.format('route_id', 'route', table_name, '1.5 year'))
cur_pg.execute(sql_mad_function.format('user_id', 'user', table_name))
cur_pg.execute(sql_mad_function.format('route_setter_id', 'rs', table_name))

# create postgres functions for mean absolute deviation computation
print("Grade range +/-: ", GRADE_OFFSET)
print("SHRINKAGE parameter: ", SHRINKAGE)
out_file_name = str(GRADE_OFFSET) + "_" + str(SHRINKAGE)

regression_climber_grade(data, table_name, intermediate_table, query,
                         features_user, features_route, features_setter, model, out_file_name,
                         features_table, features_route, features_user, threshold, GRADE_OFFSET, SHRINKAGE)

