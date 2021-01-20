import hashlib
import pandas as pd
import numpy as np

from ourlattice.accessors.lattice import Lattice, SupportFieldType
from ourlattice.accessors.facet import Facet
from ourlattice.utils.storages.storage import Storage

from typing import Callable, List, Dict

@pd.api.extensions.register_dataframe_accessor("polytope")
class Polytope(Lattice):

    support_value_index = {
        SupportFieldType.ID: 0,
        SupportFieldType.R: 1,
        SupportFieldType.W: 2,
    }
    
    def __init__(self, pandas_obj):
        super(Polytope, self).__init__(
            obj=pandas_obj,
        )

    @staticmethod
    def construct(id: str, constraints: List[Dict[str, int]]) -> pd.DataFrame:

        """
            Groups ID and R values into indices for frame.

            Return: Polytope / pd.DataFrame
        """

        try:
            _constraints = []
            for i in range(len(constraints)):
                constraint = constraints[i]
                if not SupportFieldType.B.value in constraint:
                    raise AttributeError(
                        f"Key {SupportFieldType.B.value} is required in a constraint, got: {constraint}",
                    )
                
                if not SupportFieldType.ID.value in constraint:
                    constraint[SupportFieldType.ID.value] = i

                if not SupportFieldType.R.value in constraint:
                    constraint[SupportFieldType.R.value] = None

                if not SupportFieldType.W.value in constraint:
                    constraint[SupportFieldType.W.value] = None

                _constraints.append(
                    constraint,
                )

            polytope = pd.DataFrame(_constraints).set_index([
                SupportFieldType.ID.value, 
                SupportFieldType.R.value,
                SupportFieldType.W.value,
            ]).fillna(0).sort_index()
            polytope.id = id
            polytope.hash = None
        except Exception as e:
            raise Exception(f"Could not construct polytope from constraints because of error: {e}")

        return polytope

    @staticmethod
    def _validate(obj):
        # verify there is all support fields are in object
        should_be_in_columns = [
            SupportFieldType.B.value,
        ]
        if not np.isin(should_be_in_columns, obj.columns).all():
            raise AttributeError(f"Must have {should_be_in_columns} columns in frame.")
        
        should_be_in_index = [
            SupportFieldType.R.value,
            SupportFieldType.ID.value,
            SupportFieldType.W.value,
        ]
        if not np.isin(should_be_in_index, obj.index.names).all():
            raise AttributeError(f"Must have {should_be_in_index} indexes in frame.")
            
    @property
    def variables(self):
        msk = ~np.isin(self._obj.columns, [
            SupportFieldType.B.value,
        ])
        return self._obj.columns[msk]
            
    @property
    def A(self) -> pd.DataFrame:
        return self._obj.loc[:, self.variables]
    
    @property
    def b(self) -> pd.Series:
        return self._obj[SupportFieldType.B.value]
    
    @property
    def w(self) -> pd.Series:
        return self._index_row(support_field_type=SupportFieldType.W)

    @property
    def r(self) -> pd.Series:
        return self._index_row(support_field_type=SupportFieldType.R)
    
    @property
    def id(self) -> pd.Series:
        return self._index_row(support_field_type=SupportFieldType.ID)

    def _index_row(self, support_field_type: SupportFieldType) -> pd.Series:
        r = np.array([list(k) for k in self._obj.index.values])[:, self.support_value_index[support_field_type]]
        return pd.Series(r)

    def strip(self, axis=None):
        """
            Removes rows/columns with only zero as constant. 
        """
        df = self._obj
        n_reduce = {
            None: [0, 1],
            0: [0],
            1: [1],
        }.get(axis, None)
        
        if n_reduce is None:
            raise Exception(f"Polytop cannot strip: axis {axis} does not exist.")

        for i in n_reduce:
            msk = (self.A == 0).all(axis=abs(i-1))
            if msk[msk].index.size > 0:
                df = df.drop(msk[msk].index, axis=i)

        return df
            
    def isin(self, point: dict)-> bool:
        x = pd.Series(point, index=self.variables).fillna(0)
        return (self.A.dot(x) >= self.b).all()
    
    def to_constraints(self):
        return [
            {
                **{
                    SupportFieldType.ID.value: i[0],
                    SupportFieldType.R.value: i[1],
                    SupportFieldType.W.value: i[2],
                }, 
                **r[r != 0].to_dict()
            } 
            for i, r in self._obj.iterrows()
        ]
    
    # def to_dimacscnf(self, handlers: Dict[str, Callable[[pd.Series], List[Expression]]]):

    #     """
    #         Converts into a DIMACS CNF.

    #         Return: tuple(dict, DimacsCNF)
    #     """

    #     exprs_kv = {}
    #     for (_id, rule, _), row in self._obj.iterrows():
    #         if not _id in exprs_kv:
    #             fn = handlers.get(rule, None)
    #             if not fn:
    #                 raise NotImplementedError(f"Missing handler for rule type '{rule}'")
                
    #             exprs_kv[_id] = fn(row)

    #     exprs = list(exprs_kv.values())
    #     result = expr2dimacscnf(
    #         And(*exprs).tseitin(),
    #     )

    #     return result

    # def number_of_solutions(self, sharp_sat_solver: Callable[[DimacsCNF], int], handlers: Dict[str, Callable[[pd.Series], List[Expression]]]):

    #     """
    #         Calculates the number of solutions n in this polytope.

    #         Args:
    #             sharp_sat_solver: (DimacsCNF) -> (int)
    #         Return: int
    #     """

    #     _, cnf_file_format = self.to_dimacscnf(handlers=handlers)
    #     n = sharp_sat_solver(cnf_file_format)
    #     return n

    async def save(self, _id: str, storage: Storage, compress="gzip"):
        await storage.upload_data(
            _id=_id,
            obj=self._obj,
            meta={
                "id": _id,
                "hash": self.generate_hash(),
            },
            compress=compress,
        )

    @staticmethod    
    async def load(self, _id: str, storage: Storage, decompress="gzip"):
        df = await storage.download_data(
            _id=_id,
            decompress=decompress,
        )
        return df

    def generate_hash(self) -> str:
        df = self._obj
        ignore_column_names = [
            SupportFieldType.R.value,
            SupportFieldType.B.value,
            SupportFieldType.ID.value,
        ]
        columns_sorted = sorted(list(df.columns))
        variable_header_sorted = [name for name in columns_sorted if name not in ignore_column_names]
        bts_columns = ''.join(variable_header_sorted).encode('utf-8')

        values = df[columns_sorted].values.astype(np.int16)
        indices_sorted = np.argsort([str(row) for row in values])
        bts_values = values[indices_sorted].tobytes()

        bts_box = b''.join((bts_columns, bts_values))

        return hashlib.sha256(bts_box).hexdigest()

    def falses(self, point: dict) -> pd.DataFrame:
        """
            Finding what facets that are NOT satisfied from given interpretation.

            Args:
                point: pd.Series

            Return: 
                Polytope / pd.DataFrame
        """
        point = pd.Series(point, index=self.variables).fillna(0)
        return self._obj[self.A.values.dot(point) < self.b.values]

    def trues(self, point: dict) -> pd.DataFrame:
        """
            Finding what facets that are satisfied from given interpretation.

            Args:
                point: pd.Series

            Return: 
                Polytope / pd.DataFrame
        """
        point = pd.Series(point, index=self.variables).fillna(0)
        return self._obj[self.A.values.dot(point) >= self.b.values]

    def tautologies(self) -> pd.DataFrame:

        """
            Returns tautology facets. That is facets that are satisfied, 
            no matter the interpretation.
            E.g., the facet x+y >= 0 is always true, no matter boolean
            assignment to x and y.

            Return:
                Polytope / pd.DataFrame
        """

        A, b = self.A.values, self.b.values
        A_column_sum = A.sum(axis=1)

        # Upper bound candidates
        ub_candidates_msk = (A >= b.reshape(-1,1)).all(axis=1)
        ub_taut_msk = A_column_sum >= b
        reduced_df = self._obj[ub_candidates_msk & ub_taut_msk]

        return reduced_df

    def contradictions(self) -> pd.DataFrame:

        """
            Returns contradiction facets. A contradiction
            facet is a facet that no matter the interpretation
            can be satisfied. 
            E.g., ~x & x can never be evaluated to true.

            Return:
                Polytope / pd.DataFrame
        """
        A, b = self.A.values, self.b.values

        # No variable alone is larger or equal than b
        all_less_than_b = (A < b.reshape(-1,1)).all(axis=1)
        
        # Sum of all variables is larger or equal than b
        sum_larger_msk = A.sum(axis=1) < b

        # Iff both above are true, then the facet can never be satisfied
        # and thus is a contradiction.
        contradiction_df = self._obj[all_less_than_b & sum_larger_msk]

        return contradiction_df

    def assume(self, variables: list) -> pd.DataFrame:

        """
            Assumes variables are set (has value 1) and reduces 
            polytope by removing these variables from columns.

            Args:
                variables: List[str]

            Return:
                Polytope / pd.DataFrame
        """
        # First, we need to validate input
        if not isinstance(variables, list) or not isinstance(variables[0], str):
            raise TypeError("Argument 'variables' needs to be a list of strings.")

        varisin = np.isin(variables, self.variables)
        if not varisin.all():
            raise AttributeError(f"All variables must be in this polytopes variables: {self.variables[~varisin]} is not part of this polytope.")

        A, b = self.A, self.b
        # Fix variables by:
        #   1. Sum the variables values on each row -> a = A[:, vars].sum(axis=1)
        #   2. Subtract a from b -> b' = b - a
        #   3. Remove variables from A -> A' = A.drop(variables, axis=1)
        a = A.loc[:, variables].sum(axis=1)
        _df = self._obj.drop(variables, axis=1)
        _df.loc[:, SupportFieldType.B.value] = b -a

        return _df

    def unbalanced(self) -> pd.DataFrame:

        """
            Returns a polytope of this polytopes *unbalanced* facets.
            An unbalanced facet is an unsatisfied constraint when
            no variable has been set.

            E.g., the constraint {"a": 1, "b": 0, SupportVector: 2} is unbalanced.

            Return:
                Polytope / pd.DataFrame
        """

        A, b = self.A.values, self.b.values
        msk = A.dot(np.zeros((self.variables.size))) < b
        return self._obj[msk]

    def balanced(self) -> pd.DataFrame:

        """
            Returns a polytope of this polytopes *balanced* facets.
            A balanced facet is a constraint that is satisfied when
            no variable has been set.

            E.g., the constraint {"a": 1, "b": 0, SupportVector: 0} is balanced.

            Return:
                Polytope / pd.DataFrame
        """

        A, b = self.A.values, self.b.values
        msk = A.dot(np.zeros((self.variables.size))) >= b
        return self._obj[msk]