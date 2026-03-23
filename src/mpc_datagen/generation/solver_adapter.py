import numpy as np

from abc import ABC, abstractmethod
from numpy.typing import NDArray

from acados_template import AcadosOcpSolver, AcadosOcpBatchSolver
from pkg_logger import suppress_native_output

class SolverAdapter(ABC):
    """
    Abstrakte Basisklasse (Interface) für einen Solver.
    Kapselt die Unterschiede zwischen Single- und Batch-Solvern.
    """
    
    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @abstractmethod
    def is_sqp(self) -> bool:
        pass

    @abstractmethod
    def set_x0(self, x0_batch: NDArray) -> None:
        pass

    @abstractmethod
    def set_x_guess(self, x_guess_batch: NDArray, N: int, nx: int) -> None:
        pass

    @abstractmethod
    def set_u_guess(self, u_guess_batch: NDArray, N: int, nu: int) -> None:
        pass

    @abstractmethod
    def solve(self, n: int) -> None:
        pass

    @abstractmethod
    def get_status(self, n: int) -> NDArray:
        pass

    @abstractmethod
    def get_x(self, n: int, N: int, nx: int) -> NDArray:
        pass

    @abstractmethod
    def get_u(self, n: int, N: int, nu: int) -> NDArray:
        pass

    @abstractmethod
    def get_cost(self, n: int) -> NDArray:
        pass

    @abstractmethod
    def get_time(self, n: int) -> NDArray:
        pass

    @abstractmethod
    def reset_solvers(self, done_slots: NDArray) -> None:
        pass

    @abstractmethod
    def regenerate(self, idxs: NDArray) -> None:
        pass



class AcadosSolverAdapter(SolverAdapter):
    """Adapter für den Standard AcadosOcpSolver (Batch Size = 1)."""
    
    def __init__(self, solver: AcadosOcpSolver):
        self.solver = solver

    @property
    def batch_size(self) -> int:
        return 1

    def is_sqp(self) -> bool:
        return self.solver.acados_ocp.solver_options.nlp_solver_type.lower() == "sqp"

    def set_x0(self, x0_batch: NDArray) -> None:
        x0_flat = np.asarray(x0_batch[0], dtype=float).flatten()
        self.solver.constraints_set(0, "lbx", x0_flat)
        self.solver.constraints_set(0, "ubx", x0_flat)

    def set_x_guess(self, x_guess_batch: NDArray, N: int, nx: int) -> None:
        x_guess = np.asarray(x_guess_batch[0], dtype=float).reshape(N + 1, nx)
        for k in range(N + 1):
            self.solver.set(k, "x", x_guess[k])

    def set_u_guess(self, u_guess_batch: NDArray, N: int, nu: int) -> None:
        u_guess = np.asarray(u_guess_batch[0], dtype=float).reshape(N, nu)
        for k in range(N):
            self.solver.set(k, "u", u_guess[k])

    def solve(self, n: int) -> None:
        self.solver.solve()

    def get_status(self, n: int) -> NDArray:
        return np.array([self.solver.status], dtype=int)

    def get_x(self, n: int, N: int, nx: int) -> NDArray:
        return self.solver.get_flat("x").reshape(1, N + 1, nx)

    def get_u(self, n: int, N: int, nu: int) -> NDArray:
        return self.solver.get_flat("u").reshape(1, N, nu)

    def get_cost(self, n: int) -> NDArray:
        return np.array([float(self.solver.get_cost())], dtype=float)

    def get_time(self, n: int) -> NDArray:
        return np.array([float(self.solver.get_stats("time_tot"))], dtype=float)

    def reset_solvers(self, done_slots: NDArray) -> None:
        if done_slots.size > 0:
            self.solver.reset()
    
    def regenerate(self, idxs: NDArray) -> None:
        if idxs.size > 0:
            self.solver = self._regenerate_solver(self.solver)
        
    @staticmethod
    def _regenerate_solver(solver: AcadosOcpSolver) -> AcadosOcpSolver:
        with suppress_native_output(suppress_stdout=True, suppress_stderr=True):
            return AcadosOcpSolver(
                solver.acados_ocp,
                build=False,
                check_reuse_possible=True,
            )


class AcadosBatchSolverAdapter(SolverAdapter):
    """Adapter für den AcadosOcpBatchSolver."""
    
    def __init__(self, solver: AcadosOcpBatchSolver):
        self.solver = solver

    @property
    def batch_size(self) -> int:
        return self.solver.n_batch_current

    def is_sqp(self) -> bool:
        return self.solver.ocp_solvers[0].acados_ocp.solver_options.nlp_solver_type.lower() == "sqp"

    def set_x0(self, x0_batch: NDArray) -> None:
        self.solver.constraints_set(0, "lbx", x0_batch)
        self.solver.constraints_set(0, "ubx", x0_batch)

    def set_x_guess(self, x_guess_batch: NDArray, N: int, nx: int) -> None:
        self.solver.set_flat("x", x_guess_batch.reshape((-1, nx * (N + 1))))

    def set_u_guess(self, u_guess_batch: NDArray, N: int, nu: int) -> None:
        self.solver.set_flat("u", u_guess_batch.reshape((-1, nu * N)))

    def solve(self, n: int) -> None:
        self.solver.solve(n_batch=n)

    def get_status(self, n: int) -> NDArray:
        return np.array(self.solver.status[:n], dtype=int)

    def get_x(self, n: int, N: int, nx: int) -> NDArray:
        return self.solver.get_flat("x", n_batch=n).reshape(n, N + 1, nx)

    def get_u(self, n: int, N: int, nu: int) -> NDArray:
        return self.solver.get_flat("u", n_batch=n).reshape(n, N, nu)

    def get_cost(self, n: int) -> NDArray:
        return np.array([self.solver.ocp_solvers[i].get_cost() for i in range(n)], dtype=float)

    def get_time(self, n: int) -> NDArray:
        return np.array([self.solver.ocp_solvers[i].get_stats("time_tot") for i in range(n)], dtype=float)

    def reset_solvers(self, done_slots: NDArray) -> None:
        for slot in done_slots:
            self.solver.ocp_solvers[slot].reset()

    def regenerate(self, idxs: NDArray) -> None:
        for idx in idxs:
            old_solver = self.solver.ocp_solvers[idx]
            self.solver.ocp_solvers[idx] = self._regenerate_solver(old_solver)

    @staticmethod
    def _regenerate_solver(solver: AcadosOcpSolver) -> AcadosOcpSolver:
        with suppress_native_output(suppress_stdout=True, suppress_stderr=True):
            return AcadosOcpSolver(
                solver.acados_ocp,
                build=False,
                check_reuse_possible=True,
            )