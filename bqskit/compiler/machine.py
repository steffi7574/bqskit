"""This module implements the MachineModel class."""
from __future__ import annotations

from typing import Sequence
from typing import TYPE_CHECKING
from typing import Optional

from bqskit.compiler.gateset import GateSet
from bqskit.compiler.gateset import GateSetLike
from bqskit.ir.location import CircuitLocation
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.graph import CouplingGraphLike
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_valid_radixes

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class QubitSpec:
    """
    Physical parameters of a single transmon qudit.

    freq01        : |0⟩↔|1⟩ transition frequency  (GHz)
    anharmonicity : self‑Kerr anharmonicity       (GHz)
    """
    freq01:         float
    anharmonicity:  float
    T1: Optional[float] = None 
    T2: Optional[float] = None 

class MachineModel:
    """A model of a quantum processing unit."""

    def __init__(
        self,
        num_qudits: int,
        coupling_graph: CouplingGraphLike | None = None,
        gate_set: GateSetLike | None = None,
        radixes: Sequence[int] = [],
        qubit_specs: Sequence[QubitSpec] | None = None, 
    ) -> None:
        """
        MachineModel Constructor.

        Args:
            num_qudits (int): The total number of qudits in the machine.

            coupling_graph (Iterable[tuple[int, int]] | None): A coupling
                graph describing which pairs of qudits can interact.
                Given as an undirected edge set. If left as None, then
                an all-to-all coupling graph is used as a default.
                (Default: None)

            gate_set (GateSetLike | None): The native gate set available
                on the machine. If left as None, the default gate set
                will be used. See :func:`~GateSet.default_gate_set`.

            radixes (Sequence[int]): A sequence with its length equal
                to `num_qudits`. Each element specifies the base of a
                qudit. Defaults to qubits.

            qubit_specs (Sequency[QubitSpec]): A sequence with its length 
                equal to `num_qubits`. Each element specifies the qubit 
                ground and anharmonicity frequency (GHz) and optionally T1 and T2 decay times (ns)

        Raises:
            ValueError: If `num_qudits` is nonpositive.

        Note:
            Pre-built models for many active QPUs exist in the
            :obj:`~bqskit.ext` package.
        """

        if not is_integer(num_qudits):
            raise TypeError(
                f'Expected integer num_qudits, got {type(num_qudits)}.',
            )

        if num_qudits <= 0:
            raise ValueError(f'Expected positive num_qudits, got {num_qudits}.')

        self.radixes = tuple(radixes if len(radixes) > 0 else [2] * num_qudits)

        if not is_valid_radixes(self.radixes):
            raise TypeError('Invalid qudit radixes.')

        if len(self.radixes) != num_qudits:
            raise ValueError(
                'Expected length of radixes to be equal to num_qudits:'
                ' %d != %d' % (len(self.radixes), num_qudits),
            )
        
        # --- Validate & store qubit specs 
        if qubit_specs is None:
            # default to zero
            qubit_specs = [QubitSpec(0.0, 0.0) for _ in range(num_qudits)]
        if len(qubit_specs) != num_qudits:
            raise ValueError("Need one QubitSpec per qudit.")
        self.qubit_specs: tuple[QubitSpec, ...] = tuple(qubit_specs)
        # ---
        
        if coupling_graph is None:
            coupling_graph = CouplingGraph.all_to_all(num_qudits)

        if not CouplingGraph.is_valid_coupling_graph(
                coupling_graph, num_qudits,
        ):
            raise TypeError('Invalid coupling graph, expected list of tuples')

        if gate_set is None:
            gate_set = GateSet.default_gate_set(radixes)
        else:
            gate_set = GateSet(gate_set)

        if not isinstance(gate_set, GateSet):
            raise TypeError(f'Expected GateSet, got {type(gate_set)}.')

        self.gate_set = gate_set
        self.coupling_graph = CouplingGraph(coupling_graph)
        self.num_qudits = num_qudits

    def get_locations(self, block_size: int) -> list[CircuitLocation]:
        """Return all `block_size` connected blocks of qudit indicies."""
        return self.coupling_graph.get_subgraphs_of_size(block_size)

    def is_compatible(
        self,
        circuit: Circuit,
        placement: list[int] | None = None,
    ) -> bool:
        """Check if a circuit is compatible with this model."""
        if circuit.num_qudits > self.num_qudits:
            return False

        if any(g not in self.gate_set for g in circuit.gate_set):
            return False

        if placement is None:
            placement = list(range(circuit.num_qudits))

        if any(
            (placement[e[0]], placement[e[1]]) not in self.coupling_graph
            for e in circuit.coupling_graph
        ):
            return False

        if any(
            r != self.radixes[placement[i]]
            for i, r in enumerate(circuit.radixes)
        ):
            return False

        return True

    # Helfer functions to get qubit frequencies, coupling parameters and T1, T2
    def freq01(self, q: int) -> float:
        return self.qubit_specs[q].freq01

    def anharmonicity(self, q: int) -> float:
        return self.qubit_specs[q].anharmonicity

    def T1(self, q: int) -> Optional[float]:
        return self.qubit_specs[q].T1

    def T2(self, q: int) -> Optional[float]:
        return self.qubit_specs[q].T2

    def J(self, q1: int, q2: int) -> float | None:
        return self.coupling_graph.coupling(q1, q2)

    # Overwrite qubit specs
    def set_qubit_spec(self,
        q: int,
        *,
        freq01:        float | None = None,
        anharmonicity: float | None = None,
        T1:            float | None = None,
        T2:            float | None = None,
    ) -> None:
        """
        Update the device parameters of qudit `q`. Any field left as None is **unchanged**.
        Example: > machine.set_qubit_spec(2, freq01=5.03, T1=25e-6)
        """
        if q < 0 or q >= self.num_qudits:
            raise IndexError(f"qudit index {q} out of range 0..{self.num_qudits-1}")

        # Pull the current frozen dataclass
        old = self.qubit_specs[q]

        # Fill in new values, defaulting to old ones where None
        new_spec = QubitSpec(
            freq01        = freq01        if freq01        is not None else old.freq01,
            anharmonicity = anharmonicity if anharmonicity is not None else old.anharmonicity,
            T1            = T1            if T1            is not None else old.T1,
            T2            = T2            if T2            is not None else old.T2,
        )

        # Replace the tuple entry
        specs = list(self.qubit_specs)
        specs[q] = new_spec
        self.qubit_specs = tuple(specs)