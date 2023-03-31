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
