from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold
from eis_toolkit.exceptions import InvalidCrossValidationSelected, NumberOfSplitException


class ModelPerformanceEstimation():
    def __init__(self, cross_validation_type, number_of_split):
        self.cross_validation_type = cross_validation_type
        self.number_of_split = number_of_split
        self.cross_validation_method = None

        # runs the private methods
        self.__cross_validation_methodology()

    def __cross_validation_methodology(self):
        if self.cross_validation_type is None:
            raise InvalidCrossValidationSelected

        if self.number_of_split <= 1:
            raise NumberOfSplitException
        # stratified k fold
        if self.cross_validation_type == 'LOOCV':
            self.cross_validation_method = LeaveOneOut()

        if self.cross_validation_type == 'SKFOLD':
            self.cross_validation_method = StratifiedKFold(n_splits=self.number_of_split, shuffle=True)

        if self.cross_validation_type == 'KFOLD':
            self.cross_validation_method = KFold(n_splits=self.number_of_split, shuffle=True)
