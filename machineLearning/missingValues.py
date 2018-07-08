class MissingValue:
    @staticmethod
    def replaceValuesInColumns(data, val_to_replace, replace_with_val, columns):
        tmp = data.copy()
        for col in columns:
            tmp[col] = tmp[col].map(lambda x: x if x != val_to_replace else replace_with_val)
        return tmp
