import pandas as pd


class StorageInterfaceDataTest:
    def __init__(self):
        self.project_name_ll = ["A", 'b_n', '', None]
        self.credentials_ll = ["ATYZ", '--b_n', '', None]

    def initialization(self):
        checks = [(p_name, cred) for cred in self.credentials_ll for p_name in self.project_name_ll]
        return checks

    def credentials_property(self):
        projects = [self.project_name_ll[0], self.project_name_ll[-1]]
        checks = [(p_name, cred) for cred in self.credentials_ll for p_name in projects]
        return checks

    def credentials_setter(self):
        checks = [(cred, new_cred) for cred in self.credentials_ll for new_cred in ["TYui", "-- ", None]]
        return checks

    def project_name_property(self):
        credentials = [self.credentials_ll[0], self.credentials_ll[-1]]
        checks = [(p_name, cred) for p_name in self.project_name_ll for cred in credentials]
        return checks

    def project_name_setter(self):
        checks = [(name, new_name) for name in self.project_name_ll for new_name in ["TYui", "-- ", None]]
        return checks


def gs_initialization():
    return StorageInterfaceDataTest().initialization()


def credentials_property():
    return StorageInterfaceDataTest().credentials_property()


def credentials_setter():
    return StorageInterfaceDataTest().credentials_setter()


def project_name_property():
    return StorageInterfaceDataTest().project_name_property()


def project_name_setter():
    return StorageInterfaceDataTest().project_name_setter()


def storage_to_dataframe():
    from copy import deepcopy
    df1 = pd.DataFrame(data={"A": [87, -5.1, 0], "B": ["t", "r", None]})
    df2 = pd.DataFrame(data={"A": [7, -5, 40.1], "B": ["9", "Z", "@"]})

    df3 = deepcopy(df1)
    df4 = deepcopy(df2)
    df3['Unnamed: 0'] = [0, 1, 3]
    df4['Unnamed: 0'] = [0, 1, 3]
    return [([df1, df2], ['A', 'B']), ([df3, df4], ['A', 'B'])]
