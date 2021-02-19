import pandas as pd
import numpy as np

from unittest import TestCase
from ourlattice.accessors.lattice import SupportFieldType
from ourlattice.accessors.polytope import Polytope
from ourlattice.accessors.facet import Facet

class TestPolytope(TestCase):

    def helper_get_valid_df(self):

        cnsts = [
            {
                "a": 1, 
                "b": 1, 
                "c": 1, 
                SupportFieldType.B.value: 1, 
                SupportFieldType.R.value: "exactly_one",
                SupportFieldType.ID.value: 0,    
                SupportFieldType.W.value: 10,
            },
            {
                "a": -1, 
                "b": -1, 
                "c": -1, 
                SupportFieldType.B.value: -1, 
                SupportFieldType.R.value: "exactly_one",
                SupportFieldType.ID.value: 0,    
                SupportFieldType.W.value: 10,
            },
        ]

        df = Polytope.construct(
            id="test-id",
            constraints=cnsts,
        )
        return df


    def test_will_fail_with_attribute_error_when_invalid_constraints(self):

        cnstss = [
            [
                {
                    "a":1, "b":1,
                }
            ],
            [
                {
                    "a":1, "b":1, "#id": 1,
                }
            ],
            [
                {
                    "a":1, "b":1, "#id": 1, "#r": "m"
                }
            ],
        ]

        for cnsts in cnstss:
            df = pd.DataFrame(cnsts)
            self.assertRaises(AttributeError, Polytope._validate, df)

    def test_can_fetch_r(self):

        df = self.helper_get_valid_df()
        _ = df.polytope.r

    def test_can_fetch_id(self):

        df = self.helper_get_valid_df()
        _ = df.polytope.id

    def test_can_fetch_w(self):

        df = self.helper_get_valid_df()
        _ = df.polytope.w

    def test_can_strip_polytope(self):

        df = self.helper_get_valid_df()
        df.loc[:, 'M'] = np.zeros((df.shape[0]))

        try:
            _ = df.polytope.strip()
        except Exception as e:
            self.fail(f"test_can_strip_polytope failed: {e}")

    def test_will_generate_errors_from_combination(self):

        df = self.helper_get_valid_df()
        combination = {"a": 1, "b": 1}
        errors = df.polytope.falses(combination)
        self.assertGreater(errors.shape[0], 0)

    def test_will_convert_to_constraints(self):
        
        df = self.helper_get_valid_df()
        try:
            _ = df.polytope.to_constraints()
        except Exception as e:
            self.fail(f"test_will_convert_to_constraints failed: {e}")

    def test_will_return_correct_tautologies(self):

        df = Polytope.construct(
            id=None, 
            constraints=[   
                # Tautologies
                {
                    "a": 1, 
                    "b": 1, 
                    SupportFieldType.B.value: 0, 
                    SupportFieldType.ID.value: 0,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "a": 1, 
                    "c": 1, 
                    SupportFieldType.B.value: -1,
                    SupportFieldType.ID.value: 1,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                    },
                {
                    "c": 1, 
                    "d": 3, 
                    SupportFieldType.B.value: -3,
                    SupportFieldType.ID.value: 2,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "c": -1,
                    "d": -1,
                    "e": -1,
                    SupportFieldType.B.value: -3,
                    SupportFieldType.ID.value: 3,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },

                # Not tautologies
                {
                    "x": -1, 
                    "y": -3, 
                    SupportFieldType.B.value: -3,
                    SupportFieldType.ID.value: 4,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "x": -2,
                    "y": 1,
                    "z": 1,
                    SupportFieldType.B.value: 0,
                    SupportFieldType.ID.value: 5,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "a": 1,
                    "b": 1,
                    "c": 1,
                    SupportFieldType.B.value: 1,
                    SupportFieldType.ID.value: 6,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "a": -1,
                    "b": -1,
                    "c": -1,
                    SupportFieldType.B.value: -1,
                    SupportFieldType.ID.value: 6,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
            ],
        )

        tautologies_df = df.polytope.tautologies()

        actual_index = set(tautologies_df.index)
        expected_index = set({(0,0,0), (1,0,0), (2,0,0), (3,0,0)})
        self.assertEqual(actual_index.difference(expected_index), set())

    def test_will_not_find_this_constraint_as_tautology_constraint(self):

        df = Polytope.construct(
            id=None, 
            constraints=[   
                {
                    "a": 1, 
                    "b": 1, 
                    "c": -1,
                    SupportFieldType.B.value: 0, 
                    SupportFieldType.ID.value: 0,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "a": -1,
                    "b": -1,
                    "c": 1,
                    "d": 1,
                    SupportFieldType.B.value: -1,
                    SupportFieldType.ID.value: 0,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
            ],
        )

        tautologies_df = df.polytope.tautologies()
        self.assertTrue(tautologies_df.empty)

    def test_will_return_correct_contradictions(self):

        df = Polytope.construct(
            id=None,
            constraints=[
                {
                    "a": -1,
                    "b": -1,
                    "c": -1,
                    SupportFieldType.B.value: 1,
                    SupportFieldType.ID.value: 0,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "x": 5,
                    "y": -1,
                    "z": -1,
                    SupportFieldType.B.value: 6,
                    SupportFieldType.ID.value: 1,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },

                {
                    "a": 1, 
                    "b": 1, 
                    SupportFieldType.B.value: 0, 
                    SupportFieldType.ID.value: 2,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "a": 1, 
                    "c": 1, 
                    SupportFieldType.B.value: -1,
                    SupportFieldType.ID.value: 3,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                    },
                {
                    "c": 1, 
                    "d": 3, 
                    SupportFieldType.B.value: -3,
                    SupportFieldType.ID.value: 4,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "c": -1,
                    "d": -1,
                    "e": -1,
                    SupportFieldType.B.value: -3,
                    SupportFieldType.ID.value: 5,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "x": -1, 
                    "y": -3, 
                    SupportFieldType.B.value: -3,
                    SupportFieldType.ID.value: 6,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "x": -2,
                    "y": 1,
                    "z": 1,
                    SupportFieldType.B.value: 0,
                    SupportFieldType.ID.value: 7,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "g": 1,
                    "h": 1,
                    "i": 1,
                    SupportFieldType.B.value: 3,
                    SupportFieldType.ID.value: 8,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },

                {
                    "g": -1,
                    "h": -1,
                    "i": -1,
                    SupportFieldType.B.value: 0,
                    SupportFieldType.ID.value: 9,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "g": 1,
                    "h": 1,
                    "i": 1,
                    SupportFieldType.B.value: 4,
                    SupportFieldType.ID.value: 10,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                },
                {
                    "a": -1,
                    "b": -1,
                    SupportFieldType.B.value: 0,
                    SupportFieldType.ID.value: 11,
                    SupportFieldType.R.value: 0,
                    SupportFieldType.W.value: 0,
                }
            ]
        )

        contradictions_df = df.polytope.contradictions()

        actual_index = set(contradictions_df.index)
        expected_index = {
            (0,0,0),
            (1,0,0),
            (9,0,0),
            (10,0,0),
        }
        self.assertEqual(actual_index.difference(expected_index), set())

    def test_will_remove_correct_rows_when_assuming_variables(self):

        df = Polytope.construct(
            id=None, 
            constraints=[   
                {
                    "a": 1, 
                    "b": 1, 
                    "c": 1,
                    SupportFieldType.B.value: 1, 
                    SupportFieldType.ID.value: 0,
                    SupportFieldType.R.value: "EXACTLY_ONE",
                    SupportFieldType.W.value: 0,
                },
                {
                    "a": -1, 
                    "b": -1, 
                    "c": -1,
                    SupportFieldType.B.value: -1, 
                    SupportFieldType.ID.value: 0,
                    SupportFieldType.R.value: "EXACTLY_ONE",
                    SupportFieldType.W.value: 0,
                },
                {
                    "a": -2, 
                    "d": 1, 
                    "e": 1, 
                    SupportFieldType.B.value: 0, 
                    SupportFieldType.ID.value: 1,
                    SupportFieldType.R.value: "REQUIRES_ALL",
                    SupportFieldType.W.value: 0,
                },
                {
                    "b": -1, 
                    "x": 1, 
                    "y": 1, 
                    SupportFieldType.B.value: 0, 
                    SupportFieldType.ID.value: 2,
                    SupportFieldType.R.value: "REQUIRES_EXCLUSIVELY",
                    SupportFieldType.W.value: 0,
                },
                {
                    "b": -1, 
                    "x": -1, 
                    "y": -1, 
                    SupportFieldType.B.value: -2, 
                    SupportFieldType.ID.value: 2,
                    SupportFieldType.R.value: "REQUIRES_EXCLUSIVELY",
                    SupportFieldType.W.value: 0,
                },
                {
                    "c": -1, 
                    "x": 1, 
                    "y": 1, 
                    SupportFieldType.B.value: 0, 
                    SupportFieldType.ID.value: 3,
                    SupportFieldType.R.value: "REQUIRES_EXCLUSIVELY",
                    SupportFieldType.W.value: 0,
                },
                {
                    "c": -1, 
                    "x": -1, 
                    "y": -1, 
                    SupportFieldType.B.value: -2, 
                    SupportFieldType.ID.value: 3,
                    SupportFieldType.R.value: "REQUIRES_EXCLUSIVELY",
                    SupportFieldType.W.value: 0,
                },
                
            ],
        )

        reduced_df = df.polytope.assume(
            variables=["a"],
        )

        tauts_df = reduced_df.polytope.tautologies()
        contrs_df = reduced_df.polytope.contradictions()

        self.assertEqual(tauts_df.index[0], (0,"EXACTLY_ONE",0))
        self.assertEqual(tauts_df.shape[0], 1)
        self.assertTrue(contrs_df.empty)

    def test_aggregate_weights(self):

        df = Polytope.construct(
            id=None,
            constraints=[
                {
                    "a": 1, 
                    "b": 1, 
                    "c": 1, 
                    SupportFieldType.B.value: 3, 
                    SupportFieldType.W.value: -1,
                },
                {
                    "a": 1, 
                    SupportFieldType.B.value: 1, 
                    SupportFieldType.W.value: 8,
                },
                {
                    "b": 1, 
                    SupportFieldType.B.value: 1, 
                    SupportFieldType.W.value: 9,
                },
                {
                    "c": 1, 
                    SupportFieldType.B.value: 1, 
                    SupportFieldType.W.value: 10,
                },
            ]
        )

        self.assertEqual(
            df.polytope.trues({"a": 1}).polytope.w.sum(),
            8,
        )

        self.assertEqual(
            df.polytope.trues({"a": 1, "b": 1}).polytope.w.sum(),
            8+9,
        )

        self.assertEqual(
            df.polytope.trues({"a": 1, "b": 1, "c": 1}).polytope.w.sum(),
            8+9+10-1,
        )

    def test_to_strip(self):
        df = Polytope.construct(
            id=None, 
            constraints=[   
                {
                    "a": 1, 
                    "b": 1, 
                    "c": 0,
                    SupportFieldType.B.value: 1, 
                    SupportFieldType.ID.value: 0,
                    SupportFieldType.R.value: "EXACTLY_ONE",
                    SupportFieldType.W.value: 0,
                },
                {
                    "a": -1, 
                    "b": -1, 
                    "c": 0,
                    SupportFieldType.B.value: -1, 
                    SupportFieldType.ID.value: 0,
                    SupportFieldType.R.value: "EXACTLY_ONE",
                    SupportFieldType.W.value: 0,
                },
                {
                    "a": -1, 
                    "b": 1, 
                    "c": 0,
                    SupportFieldType.B.value: 0, 
                    SupportFieldType.ID.value: 1,
                    SupportFieldType.R.value: "REQUIRES_ALL",
                    SupportFieldType.W.value: 0,
                },
                {
                    "a": 0, 
                    "b": 0, 
                    "c": 0,
                    SupportFieldType.B.value: 0, 
                    SupportFieldType.ID.value: 2,
                    SupportFieldType.R.value: "EMPTY",
                    SupportFieldType.W.value: 0,
                },
            ],
        )

        stripped_full_df = df.polytope.strip()
        self.assertIn((0, "EXACTLY_ONE", 0), stripped_full_df.index)
        self.assertIn((1, "REQUIRES_ALL", 0), stripped_full_df.index)
        self.assertNotIn((2, "EMPTY", 0), stripped_full_df.index)
        self.assertNotIn("c", stripped_full_df.columns)
        
        stripped_row_df = df.polytope.strip(axis=0)
        self.assertIn((0, "EXACTLY_ONE", 0), stripped_row_df.index)
        self.assertIn((1, "REQUIRES_ALL", 0), stripped_row_df.index)
        self.assertNotIn((2, "EMPTY", 0), stripped_row_df.index)
        self.assertIn("c", stripped_row_df.columns)

        stripped_clm_df = df.polytope.strip(axis=1)
        self.assertIn((0, "EXACTLY_ONE", 0), stripped_clm_df.index)
        self.assertIn((1, "REQUIRES_ALL", 0), stripped_clm_df.index)
        self.assertIn((2, "EMPTY", 0), stripped_clm_df.index)
        self.assertNotIn("c", stripped_clm_df.columns)

        # Test to strip nothing
        df = self.helper_get_valid_df()
        stripped = df.polytope.strip()
        self.assertEqual(df.shape, stripped.shape)

        # Test to strip with invalid axis
        self.assertRaises(
            Exception,
            df.polytope.strip,
            kwargs={
                "axis": "invalid_axis",
            }
        )
