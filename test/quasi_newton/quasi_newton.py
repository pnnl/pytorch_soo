#!/usr/bin/env python3
"""
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
import unittest
import torch
from pytorch_sso import matrix_free_operators
from pytorch_sso import matrix_operators

NUM_TEST_CASES = 1
PROBLEM_DIMENSION = 3
RAND_SCALE = 100.0
TEST_TOLERANCE = 1e-6

DTYPE = torch.float64


class SR1Operators(unittest.TestCase):
    def same_vec(self, vec1, vec2, test_tol=TEST_TOLERANCE):
        diff = vec1 - vec2
        norm = torch.norm(diff)
        self.assertLessEqual(norm, test_tol)

    def test_sr1_identity_recon(self):
        B0p = lambda p: p
        mf_sr1 = matrix_free_operators.SymmetricRankOne(B0p)
        sr1 = matrix_operators.SymmetricRankOne(
            torch.eye(PROBLEM_DIMENSION, dtype=DTYPE)
        )
        sr1_recon = mf_sr1.construct_full_matrix(PROBLEM_DIMENSION, dtype=DTYPE)
        self.same_vec(sr1.matrix, sr1_recon)

    def test_sr1_one_update_recon(self):
        B0p = lambda p: p
        B0 = torch.eye(PROBLEM_DIMENSION, dtype=DTYPE)
        mf_sr1 = matrix_free_operators.SymmetricRankOne(B0p)
        sr1 = matrix_operators.SymmetricRankOne(B0)

        delta_x = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
        grad_f_k = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
        grad_f_k_plus_one = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)

        mf_Bk_delta_x = mf_sr1 * delta_x
        Bk_delta_x = sr1 * delta_x
        self.same_vec(mf_Bk_delta_x, Bk_delta_x)

        mf_sr1.update(grad_f_k, grad_f_k_plus_one, delta_x)
        sr1.update(grad_f_k, grad_f_k_plus_one, delta_x)

        sr1_recon = mf_sr1.construct_full_matrix(PROBLEM_DIMENSION, dtype=DTYPE)
        self.same_vec(sr1.matrix.flatten(), sr1_recon.flatten())

    def test_sr1_two_update_every_step(self):
        steps = []
        for _ in range(2):
            delta_x = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
            grad_f_k = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
            grad_f_k_plus_one = RAND_SCALE * torch.rand(
                (PROBLEM_DIMENSION,), dtype=DTYPE
            )

            test_vec = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)

            step = (delta_x, grad_f_k, grad_f_k_plus_one, test_vec)
            steps.append(step)

        # Full matrix
        B = torch.eye(PROBLEM_DIMENSION, dtype=DTYPE)
        m_rslts = []
        init_vec = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
        m_rslts.append(torch.matmul(B, init_vec))

        for step in steps:
            delta_x = step[0]
            grad_f_k = step[1]
            grad_f_k_plus_one = step[2]
            test_vec = step[3]

            yk = grad_f_k_plus_one - grad_f_k
            Bk_delta_x = torch.matmul(B, delta_x)
            vk = yk - Bk_delta_x
            num = torch.outer(vk, vk)
            den = torch.dot(vk, delta_x)
            B = num / den
            rslt = torch.matmul(B, test_vec)
            m_rslts.append(rslt)

        # MF
        B0p = lambda p: p
        mf_B = matrix_free_operators.SymmetricRankOne(B0p)
        mf_rslts = []
        mf_rslts.append(mf_B * init_vec)

        for step in steps:
            test_vec = step[3]
            mf_B.update(*step[:3])
            mf_rslts.append(mf_B * test_vec)

    def test_sr1_two_update_recon(self):
        B0p = lambda p: p
        mf_sr1 = matrix_free_operators.SymmetricRankOne(B0p)
        sr1 = matrix_operators.SymmetricRankOne(
            torch.eye(PROBLEM_DIMENSION, dtype=DTYPE)
        )
        sr1_recon = mf_sr1.construct_full_matrix(PROBLEM_DIMENSION, dtype=DTYPE)

        for _ in range(2):
            delta_x = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
            grad_f_k = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
            grad_f_k_plus_one = RAND_SCALE * torch.rand(
                (PROBLEM_DIMENSION,), dtype=DTYPE
            )

            test_vec = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)

            # Let's check the inner results manually
            yk = grad_f_k_plus_one - grad_f_k
            Bk_delta_x = sr1 * delta_x
            mf_Bk_delta_x = mf_sr1 * delta_x
            vk = yk - Bk_delta_x
            mf_vk = yk - mf_Bk_delta_x
            num = torch.outer(vk, vk)
            mf_num = torch.outer(mf_vk, mf_vk)

            num_prod = torch.inner(num, test_vec)
            mf_num_prod = torch.inner(mf_num, test_vec)
            mf_num_prod_inner = vk * torch.dot(vk, test_vec)
            self.same_vec(num_prod, mf_num_prod)
            self.same_vec(mf_num_prod, mf_num_prod_inner)
            self.same_vec(num_prod, mf_num_prod_inner)
            den = torch.dot(vk, delta_x)
            mf_den = torch.dot(mf_vk, delta_x)

            self.same_vec(den, mf_den)

            exact_rslt = num_prod / den
            mf_rslt = mf_num_prod_inner / mf_den
            self.same_vec(num / den, mf_num / mf_den)
            self.same_vec(exact_rslt, mf_rslt)
            self.assertEqual(exact_rslt.shape[0], mf_rslt.shape[0])

            self.assertLessEqual(torch.norm(mf_Bk_delta_x - Bk_delta_x), TEST_TOLERANCE)

            mf_sr1.update(grad_f_k, grad_f_k_plus_one, delta_x)
            sr1.update(grad_f_k, grad_f_k_plus_one, delta_x)
            sr1_recon = mf_sr1.construct_full_matrix(PROBLEM_DIMENSION, dtype=DTYPE)
            pass

        sr1_recon = mf_sr1.construct_full_matrix(PROBLEM_DIMENSION, dtype=DTYPE)
        with self.subTest(msg="Flattened matrix"):
            self.same_vec(sr1_recon.flatten(), sr1.matrix.flatten())
        with self.subTest(msg="Full matrix"):
            rslt = sr1.matrix - sr1_recon
            self.assertLessEqual(torch.norm(rslt), TEST_TOLERANCE)

    def test_sr1_identity(self):
        B0p = lambda p: p
        mf_sr1 = matrix_free_operators.SymmetricRankOne(B0p)
        sr1 = matrix_operators.SymmetricRankOne(
            torch.eye(PROBLEM_DIMENSION, dtype=DTYPE)
        )
        for _ in range(NUM_TEST_CASES):
            with self.subTest():
                vec = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                prod = sr1 * vec
                mf_prod = mf_sr1 * vec
                rslt = torch.norm(prod - mf_prod)
                self.assertLessEqual(rslt, TEST_TOLERANCE)

    def test_sr1_one_update(self):
        B0p = lambda p: p
        mf_sr1 = matrix_free_operators.SymmetricRankOne(B0p)
        sr1 = matrix_operators.SymmetricRankOne(
            torch.eye(PROBLEM_DIMENSION, dtype=DTYPE)
        )

        for _ in range(NUM_TEST_CASES):
            with self.subTest():
                mf_sr1.reset()
                sr1.reset()

                delta_x = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                grad_f_k = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                grad_f_k_plus_one = RAND_SCALE * torch.rand(
                    (PROBLEM_DIMENSION,), dtype=DTYPE
                )

                mf_sr1.update(grad_f_k, grad_f_k_plus_one, delta_x)
                sr1.update(grad_f_k, grad_f_k_plus_one, delta_x)

                vec = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                prod = sr1 * vec
                mf_prod = mf_sr1 * vec
                rslt = torch.norm(prod - mf_prod)
                self.assertLessEqual(rslt, TEST_TOLERANCE)

    def test_sr1_two_update(self):
        B0p = lambda p: p
        mf_sr1 = matrix_free_operators.SymmetricRankOne(B0p)
        sr1 = matrix_operators.SymmetricRankOne(
            torch.eye(PROBLEM_DIMENSION, dtype=DTYPE)
        )

        for _ in range(NUM_TEST_CASES):
            with self.subTest():
                mf_sr1.reset()
                sr1.reset()

                for _ in range(2):
                    delta_x = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                    grad_f_k = RAND_SCALE * torch.rand(
                        (PROBLEM_DIMENSION,), dtype=DTYPE
                    )
                    grad_f_k_plus_one = RAND_SCALE * torch.rand(
                        (PROBLEM_DIMENSION,), dtype=DTYPE
                    )

                    mf_sr1.update(grad_f_k, grad_f_k_plus_one, delta_x)
                    sr1.update(grad_f_k, grad_f_k_plus_one, delta_x)

                vec = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                prod = sr1 * vec
                mf_prod = mf_sr1 * vec
                rslt = torch.norm(prod - mf_prod)
                self.assertLessEqual(rslt, TEST_TOLERANCE)


class BroydenOperators(unittest.TestCase):
    def same_vec(self, vec1, vec2, test_tol=TEST_TOLERANCE):
        diff = vec1 - vec2
        norm = torch.norm(diff)
        self.assertLessEqual(norm, test_tol)

    def test_broyden_identity(self):
        B0p = lambda p: p
        mf_broyden = matrix_free_operators.Broyden(B0p)
        broyden = matrix_operators.Broyden(torch.eye(PROBLEM_DIMENSION, dtype=DTYPE))
        for _ in range(NUM_TEST_CASES):
            with self.subTest():
                vec = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                prod = broyden * vec
                mf_prod = mf_broyden * vec
                rslt = torch.norm(prod - mf_prod)
                self.assertLessEqual(rslt, TEST_TOLERANCE)

    def test_broyden_identity_recon(self):
        B0p = lambda p: p
        mf_broyden = matrix_free_operators.Broyden(B0p)
        broyden = matrix_operators.Broyden(torch.eye(PROBLEM_DIMENSION, dtype=DTYPE))
        broyden_recon = mf_broyden.construct_full_matrix(PROBLEM_DIMENSION, dtype=DTYPE)
        self.same_vec(broyden.matrix, broyden_recon)

    def test_broyden_one_update_recon(self):
        B0p = lambda p: p
        B0 = torch.eye(PROBLEM_DIMENSION, dtype=DTYPE)
        mf_broyden = matrix_free_operators.Broyden(B0p)
        broyden = matrix_operators.Broyden(B0)

        delta_x = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
        grad_f_k = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
        grad_f_k_plus_one = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)

        mf_Bk_delta_x = mf_broyden * delta_x
        Bk_delta_x = broyden * delta_x
        self.same_vec(mf_Bk_delta_x, Bk_delta_x)

        mf_broyden.update(grad_f_k, grad_f_k_plus_one, delta_x)
        broyden.update(grad_f_k, grad_f_k_plus_one, delta_x)

        mf_broyden = mf_broyden.construct_full_matrix(PROBLEM_DIMENSION, dtype=DTYPE)
        self.same_vec(broyden.matrix, mf_broyden)

    def test_broyden_two_update_recon(self):
        B0p = lambda p: p
        B0 = torch.eye(PROBLEM_DIMENSION, dtype=DTYPE)
        mf_broyden = matrix_free_operators.Broyden(B0p)
        broyden = matrix_operators.Broyden(B0)

        for _ in range(2):
            delta_x = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
            grad_f_k = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
            grad_f_k_plus_one = RAND_SCALE * torch.rand(
                (PROBLEM_DIMENSION,), dtype=DTYPE
            )

            mf_Bk_delta_x = mf_broyden * delta_x
            Bk_delta_x = broyden * delta_x
            self.same_vec(mf_Bk_delta_x, Bk_delta_x)

            mf_broyden.update(grad_f_k, grad_f_k_plus_one, delta_x)
            broyden.update(grad_f_k, grad_f_k_plus_one, delta_x)

        mf_broyden = mf_broyden.construct_full_matrix(PROBLEM_DIMENSION, dtype=DTYPE)
        self.same_vec(broyden.matrix, mf_broyden)

    def test_broyden_one_update(self):
        B0p = lambda p: p
        mf_broyden = matrix_free_operators.Broyden(B0p)
        broyden = matrix_operators.Broyden(torch.eye(PROBLEM_DIMENSION, dtype=DTYPE))

        for _ in range(NUM_TEST_CASES):
            with self.subTest():
                mf_broyden.reset()
                broyden.reset()

                delta_x = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                grad_f_k = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                grad_f_k_plus_one = RAND_SCALE * torch.rand(
                    (PROBLEM_DIMENSION,), dtype=DTYPE
                )

                mf_broyden.update(grad_f_k, grad_f_k_plus_one, delta_x)
                broyden.update(grad_f_k, grad_f_k_plus_one, delta_x)

                vec = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                prod = broyden * vec
                mf_prod = mf_broyden * vec
                rslt = torch.norm(prod - mf_prod)
                self.assertLessEqual(rslt, TEST_TOLERANCE)

    def test_broyden_two_update(self):
        B0p = lambda p: p
        mf_broyden = matrix_free_operators.Broyden(B0p)
        broyden = matrix_operators.Broyden(torch.eye(PROBLEM_DIMENSION, dtype=DTYPE))

        for _ in range(NUM_TEST_CASES):
            with self.subTest():
                mf_broyden.reset()
                broyden.reset()

                for _ in range(2):
                    delta_x = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                    grad_f_k = RAND_SCALE * torch.rand(
                        (PROBLEM_DIMENSION,), dtype=DTYPE
                    )
                    grad_f_k_plus_one = RAND_SCALE * torch.rand(
                        (PROBLEM_DIMENSION,), dtype=DTYPE
                    )

                    mf_broyden.update(grad_f_k, grad_f_k_plus_one, delta_x)
                    broyden.update(grad_f_k, grad_f_k_plus_one, delta_x)

                vec = RAND_SCALE * torch.rand((PROBLEM_DIMENSION,), dtype=DTYPE)
                prod = broyden * vec
                mf_prod = mf_broyden * vec
                rslt = torch.norm(prod - mf_prod)
                self.assertLessEqual(rslt, TEST_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
