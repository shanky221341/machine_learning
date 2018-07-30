class features:
    raw_columns = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
                   'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                   'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
                   'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
                   'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'HOUR_APPR_PROCESS_START',
                   'DAYS_LAST_PHONE_CHANGE',
                   'LIVINGAPARTMENTS_AVG',
                   'AMT_REQ_CREDIT_BUREAU_DAY',
                   'FLOORSMAX_MODE',
                   'COMMONAREA_MEDI',
                   'NONLIVINGAREA_AVG',
                   'NONLIVINGAPARTMENTS_MODE',
                   'ELEVATORS_MEDI',
                   'COMMONAREA_MEDI',
                   'FLOORSMAX_MODE',
                   'YEARS_BUILD_MEDI',
                   'APARTMENTS_MEDI',
                   'NONLIVINGAREA_MEDI',
                   'LANDAREA_MEDI',
                   'NONLIVINGAPARTMENTS_MODE',
                   'LIVINGAPARTMENTS_MODE',
                   'YEARS_BUILD_MODE',
                   'LANDAREA_MODE',
                   'FLOORSMIN_AVG',
                   'DEF_60_CNT_SOCIAL_CIRCLE',
                   'OBS_60_CNT_SOCIAL_CIRCLE',
                   'YEARS_BEGINEXPLUATATION_MODE',
                   'NONLIVINGAPARTMENTS_MEDI',
                   'NONLIVINGAREA_AVG',
                   'YEARS_BUILD_MODE',
                   'YEARS_BUILD_AVG',
                   'AMT_REQ_CREDIT_BUREAU_YEAR',
                   'APARTMENTS_MODE',
                   'COMMONAREA_MODE',
                   'LIVINGAREA_AVG',
                   'EXT_SOURCE_2',
                   'LANDAREA_AVG',
                   'ENTRANCES_MODE',
                   'FLOORSMAX_MEDI',
                   'NONLIVINGAPARTMENTS_AVG',
                   'APARTMENTS_AVG',
                   'YEARS_BEGINEXPLUATATION_MEDI',
                   'TOTALAREA_MODE',
                   'EXT_SOURCE_3',
                   'OBS_30_CNT_SOCIAL_CIRCLE',
                   'BASEMENTAREA_AVG',
                   'FLOORSMAX_MODE',
                   'DEF_30_CNT_SOCIAL_CIRCLE',
                   'FLOORSMIN_MODE',
                   'LIVINGAPARTMENTS_MEDI',
                   'BASEMENTAREA_MODE',
                   'AMT_REQ_CREDIT_BUREAU_HOUR',
                   'ELEVATORS_AVG',
                   'ENTRANCES_MEDI',
                   'LIVINGAREA_MODE',
                   'EXT_SOURCE_1',
                   'ENTRANCES_AVG',
                   'FLOORSMIN_MEDI',
                   'YEARS_BEGINEXPLUATATION_AVG',
                   'FLOORSMAX_AVG',
                   'CNT_FAM_MEMBERS',
                   'BASEMENTAREA_MEDI',
                   'LIVINGAREA_MEDI',
                   'AMT_REQ_CREDIT_BUREAU_WEEK',
                   'AMT_REQ_CREDIT_BUREAU_MON',
                   'AMT_REQ_CREDIT_BUREAU_QRT',
                   'ELEVATORS_MODE',
                   'COMMONAREA_AVG',
                   'NONLIVINGAREA_MODE',
                   'BASEMENTAREA_AVG',

                   'FLAG_CONT_MOBILE',
                   'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
                   'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
                   'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
                   'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
                   'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                   'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_EMAIL', 'FLAG_EMP_PHONE',
                   'FLAG_PHONE', 'FLAG_WORK_PHONE', 'LIVE_CITY_NOT_WORK_CITY',
                   'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
                   'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION',
                   'REG_REGION_NOT_WORK_REGION'
                   ]
    raw_columns = list(set(raw_columns))
    miss_columns = raw_columns

    freq_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                    'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'NAME_TYPE_SUITE',
                    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                    'FLAG_MOBIL',
                    'FLAG_EMP_PHONE',
                    'FLAG_WORK_PHONE',
                    'FLAG_CONT_MOBILE',
                    'FLAG_PHONE',
                    'FLAG_EMAIL',
                    'FLAG_DOCUMENT_2',
                    'FLAG_DOCUMENT_3',
                    'FLAG_DOCUMENT_4',
                    'FLAG_DOCUMENT_5',
                    'FLAG_DOCUMENT_6',
                    'FLAG_DOCUMENT_7',
                    'FLAG_DOCUMENT_8',
                    'FLAG_DOCUMENT_9',
                    'FLAG_DOCUMENT_10',
                    'FLAG_DOCUMENT_11',
                    'FLAG_DOCUMENT_12',
                    'FLAG_DOCUMENT_13',
                    'FLAG_DOCUMENT_14',
                    'FLAG_DOCUMENT_15',
                    'FLAG_DOCUMENT_16',
                    'FLAG_DOCUMENT_17',
                    'FLAG_DOCUMENT_18',
                    'FLAG_DOCUMENT_19',
                    'FLAG_DOCUMENT_20',
                    'FLAG_DOCUMENT_21',
                    'WALLSMATERIAL_MODE',
                    'FONDKAPREMONT_MODE',
                    'OCCUPATION_TYPE',
                    'HOUSETYPE_MODE',
                    'EMERGENCYSTATE_MODE',
                    'WEEKDAY_APPR_PROCESS_START',
                    'ORGANIZATION_TYPE',
                    'REGION_RATING_CLIENT_W_CITY',
                    'REG_REGION_NOT_WORK_REGION',
                    'REGION_RATING_CLIENT',
                    'REG_CITY_NOT_WORK_CITY',
                    'LIVE_CITY_NOT_WORK_CITY',
                    'REG_REGION_NOT_LIVE_REGION',
                    'LIVE_REGION_NOT_WORK_REGION'
                    ]

    tmp_removal = ['EMERGENCYSTATE_MODE',
                   'FONDKAPREMONT_MODE',
                   'HOUSETYPE_MODE',
                   'NAME_TYPE_SUITE',
                   'OCCUPATION_TYPE',
                   'WALLSMATERIAL_MODE']

    freq_columns = list(set(freq_columns) - set(tmp_removal))

    one_hot_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                       'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',
                       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                       'FLAG_MOBIL',
                       'NAME_INCOME_TYPE',
                       'FLAG_EMP_PHONE',
                       'FLAG_WORK_PHONE',
                       'FLAG_CONT_MOBILE',
                       'FLAG_PHONE',
                       'FLAG_EMAIL',
                       'FLAG_DOCUMENT_2',
                       'FLAG_DOCUMENT_3',
                       'FLAG_DOCUMENT_4',
                       'FLAG_DOCUMENT_5',
                       'FLAG_DOCUMENT_6',
                       'FLAG_DOCUMENT_7',
                       'FLAG_DOCUMENT_8',
                       'FLAG_DOCUMENT_9',
                       'FLAG_DOCUMENT_10',
                       'FLAG_DOCUMENT_11',
                       'FLAG_DOCUMENT_12',
                       'FLAG_DOCUMENT_13',
                       'FLAG_DOCUMENT_14',
                       'FLAG_DOCUMENT_15',
                       'FLAG_DOCUMENT_16',
                       'FLAG_DOCUMENT_17',
                       'FLAG_DOCUMENT_18',
                       'FLAG_DOCUMENT_19',
                       'FLAG_DOCUMENT_20',
                       'FLAG_DOCUMENT_21',
                       'WALLSMATERIAL_MODE',
                       'FONDKAPREMONT_MODE',
                       'OCCUPATION_TYPE',
                       'HOUSETYPE_MODE',
                       'EMERGENCYSTATE_MODE',
                       'WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE',
                       'REGION_RATING_CLIENT_W_CITY',
                       'REG_REGION_NOT_WORK_REGION',
                       'REGION_RATING_CLIENT',
                       'REG_CITY_NOT_WORK_CITY',
                       'LIVE_CITY_NOT_WORK_CITY',
                       'REG_REGION_NOT_LIVE_REGION',
                       'LIVE_REGION_NOT_WORK_REGION',
                       'REG_CITY_NOT_LIVE_CITY'
                       ]

    one_hot_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                       'FLAG_OWN_REALTY', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE'
                       ]

    # one_hot_columns = ['CODE_GENDER', 'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_10',
    #                    'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
    #                    'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
    #                    'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
    #                    'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
    #                    'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
    #                    'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
    #                    'FLAG_DOCUMENT_9', 'FLAG_EMAIL', 'FLAG_EMP_PHONE', 'FLAG_MOBIL',
    #                    'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_PHONE', 'FLAG_WORK_PHONE',
    #                    'LIVE_CITY_NOT_WORK_CITY', 'LIVE_REGION_NOT_WORK_REGION',
    #                    'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
    #                    'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 'ORGANIZATION_TYPE',
    #                    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
    #                    'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
    #                    'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
    #                    'WEEKDAY_APPR_PROCESS_START', 'OCCUPATION_TYPE']


cat_miss_columns = ['OCCUPATION_TYPE']
