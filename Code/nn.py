"""
Neuronales Netz from Scratch – nach 4AHITS_NN_from_scratch.pdf (Part 1).

Aufbau:  Neuron  ->  Layer (Liste von Neuronen)  ->  NeuralNetwork (Liste von Layern)
Training: Stochastic Gradient Descent (ein Sample pro Schritt), Kostenfunktion MSE  J = 1/2 (A - Y)^2
Nur Python-Standardbibliothek.
"""

import json
import math
import random


# ---------------------------------------------------------------------------
# Aktivierungsfunktionen
# Jede Funktion hat ihre Ableitung, ausgedrückt über den OUTPUT a (nicht über z).
# Beispiel Sigmoid (PDF Seite 4):  dA/dZ = A * (1 - A)
# ---------------------------------------------------------------------------

def sigmoid(z: float) -> float:
    """Sigmoid: 1 / (1 + e^-z). Gegen Overflow bei sehr großen |z| abgesichert."""
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(a: float) -> float:
    """Ableitung der Sigmoid, berechnet aus dem Output a."""
    return a * (1.0 - a)


def tanh_derivative(a: float) -> float:
    """Ableitung von tanh, berechnet aus dem Output a."""
    return 1.0 - a * a


def relu(z: float) -> float:
    """ReLU: max(0, z)."""
    return max(0.0, z)


def relu_derivative(a: float) -> float:
    """Ableitung von ReLU: 1 wenn aktiv, sonst 0."""
    return 1.0 if a > 0 else 0.0


# Name -> (Funktion, Ableitung). Über den Namen wird die Aktivierung auch im JSON gespeichert.
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (math.tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
}


# ---------------------------------------------------------------------------
# Neuron
# ---------------------------------------------------------------------------

class Neuron:
    """Ein einzelnes Neuron: speichert Gewichte, Bias und seinen letzten Input/Output."""

    def __init__(self, num_inputs: int, activation: str = "sigmoid"):
        # Zufällige Startwerte (PDF: "the weights are randomly initialized")
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(num_inputs)]
        self.bias = random.uniform(-1.0, 1.0)
        self.activation = activation
        self.last_inputs: list[float] = []
        self.last_output = 0.0

    def activate(self, inputs: list[float]) -> float:
        """Forward: output = activation(dot(inputs, weights) + bias)."""
        func, _ = ACTIVATIONS[self.activation]
        z = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
        self.last_inputs = inputs
        self.last_output = func(z)
        return self.last_output

    def delta(self, error: float) -> float:
        """
        delta = dJ/dZ = dJ/dA * dA/dZ.
        'error' ist dJ/dA, das von der nächsten Schicht (oder der Kostenfunktion) kommt.
        """
        _, derivative = ACTIVATIONS[self.activation]
        return error * derivative(self.last_output)

    def update(self, delta: float, learning_rate: float) -> None:
        """
        Gradient Descent (PDF Seite 6):  W_new = W_old - alpha * dJ/dW
        dJ/dW_i = delta * input_i   (weil dZ/dW_i = input_i)
        dJ/db   = delta             (weil dZ/db = 1)
        """
        for i, x in enumerate(self.last_inputs):
            self.weights[i] -= learning_rate * delta * x
        self.bias -= learning_rate * delta


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class Layer:
    """Eine Schicht aus Neuronen, die alle denselben Input bekommen."""

    def __init__(self, num_neurons: int, num_inputs: int, activation: str = "sigmoid"):
        self.neurons = [Neuron(num_inputs, activation) for _ in range(num_neurons)]

    def forward(self, inputs: list[float]) -> list[float]:
        """Berechnet den Output jedes Neurons dieser Schicht."""
        return [neuron.activate(inputs) for neuron in self.neurons]

    def backward(self, errors: list[float], learning_rate: float) -> list[float]:
        """
        Backpropagation für diese Schicht.
        errors[j] = dJ/dA des j-ten Neurons.
        Rückgabe: dJ/dA für die VORHERIGE Schicht (Kettenregel, dZ/dA_prev = W).
        """
        deltas = [neuron.delta(err) for neuron, err in zip(self.neurons, errors)]

        # Fehler für die vorherige Schicht MUSS mit den alten Gewichten berechnet werden,
        # also bevor wir die Gewichte updaten.
        num_inputs = len(self.neurons[0].weights)
        prev_errors = [0.0] * num_inputs
        for neuron, d in zip(self.neurons, deltas):
            for i in range(num_inputs):
                prev_errors[i] += d * neuron.weights[i]

        for neuron, d in zip(self.neurons, deltas):
            neuron.update(d, learning_rate)

        return prev_errors


# ---------------------------------------------------------------------------
# NeuralNetwork
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    Das komplette Netz.
    layer_sizes = [Inputs, Hidden..., Outputs], z. B. [2, 2, 1] für XOR.
    Die Input-Schicht hat keine Neuronen, sie gibt nur die Werte weiter.
    """

    def __init__(self, layer_sizes: list[int], activation: str = "sigmoid"):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.layers = [
            Layer(layer_sizes[i + 1], layer_sizes[i], activation)
            for i in range(len(layer_sizes) - 1)
        ]

    # --- Forward -----------------------------------------------------------

    def forward(self, inputs: list[float]) -> list[float]:
        """Schickt den Input durch alle Schichten (A[0] -> A[1] -> ... -> A[L])."""
        values = inputs
        for layer in self.layers:
            values = layer.forward(values)
        return values

    def predict(self, inputs: list[float]) -> list[float]:
        """Inferenz: nur Forward Pass, liefert die Ausgabe der letzten Schicht."""
        return self.forward(inputs)

    # --- Backward & Training -----------------------------------------------

    def backward(self, targets: list[float], learning_rate: float) -> None:
        """
        Startet bei der Kostenfunktion: dJ/dA[L] = A[L] - Y  (PDF Seite 4)
        und gibt den Fehler Schicht für Schicht nach hinten weiter.
        """
        outputs = [n.last_output for n in self.layers[-1].neurons]
        errors = [a - y for a, y in zip(outputs, targets)]
        for layer in reversed(self.layers):
            errors = layer.backward(errors, learning_rate)

    def train(self, inputs: list[list[float]], targets: list[list[float]],
              epochs: int = 20000, learning_rate: float = 0.1,
              log_every: int = 2000, on_epoch=None) -> list[float]:
        """
        Trainiert mit SGD: pro Epoche wird jedes Sample einzeln (in zufälliger Reihenfolge)
        vorwärts und rückwärts geschickt. Gibt den Verlauf des mittleren Loss zurück.
        on_epoch(epoch, loss) wird optional nach jeder Epoche aufgerufen (z. B. für Schnappschüsse).
        """
        samples = list(zip(inputs, targets))
        history = []
        for epoch in range(1, epochs + 1):
            random.shuffle(samples)
            total_loss = 0.0
            for x, y in samples:
                output = self.forward(x)
                total_loss += sum(0.5 * (a - t) ** 2 for a, t in zip(output, y))
                self.backward(y, learning_rate)
            mean_loss = total_loss / len(samples)
            history.append(mean_loss)
            if on_epoch:
                on_epoch(epoch, mean_loss)
            if log_every and (epoch % log_every == 0 or epoch == 1):
                print(f"Epoch {epoch:>6}  Loss {mean_loss:.6f}")
        return history

    # --- Export / Import ---------------------------------------------------

    def to_dict(self) -> dict:
        """Struktur + alle Gewichte als Dictionary (Basis für JSON und JS-Export)."""
        return {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation,
            "layers": [
                [{"weights": n.weights, "bias": n.bias} for n in layer.neurons]
                for layer in self.layers
            ],
        }

    def dump_json(self, path: str) -> None:
        """Speichert das Netz als JSON-Datei."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NeuralNetwork":
        """Baut ein Netz aus einer JSON-Datei wieder auf."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        net = cls(data["layer_sizes"], data["activation"])
        for layer, layer_data in zip(net.layers, data["layers"]):
            for neuron, neuron_data in zip(layer.neurons, layer_data):
                neuron.weights = list(neuron_data["weights"])
                neuron.bias = neuron_data["bias"]
        return net

    def dump_js(self, path: str, var_name: str = "NETWORK") -> None:
        """
        Speichert das Netz als JavaScript-Datei (const NETWORK = {...}).
        Grund: Eine lokal geöffnete HTML-Seite darf keine JSON per fetch() laden,
        eine <script>-Datei aber schon.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"const {var_name} = ")
            json.dump(self.to_dict(), f, indent=2)
            f.write(";\n")

    def dump_txt(self, path: str) -> None:
        """
        Menschenlesbarer Export (PDF Part 1, Step 5). Das Format ist so gewählt,
        dass es später auch ein C-Programm Zeile für Zeile einlesen kann (Part 4).
        """
        lines = [
            "# Neural Network Export",
            f"activation {self.activation}",
            f"inputs {self.layer_sizes[0]}",
            f"layers {len(self.layers)}",
            "",
        ]
        for l, layer in enumerate(self.layers, start=1):
            lines.append(f"layer {l} neurons {len(layer.neurons)}")
            for n, neuron in enumerate(layer.neurons, start=1):
                weights = " ".join(f"{w:.10f}" for w in neuron.weights)
                lines.append(f"  neuron {n} bias {neuron.bias:.10f} weights {weights}")
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def dump_initfile(self, path: str, prefix: str = "NN") -> None:
        """
        Erzeugt ein C-Header-File (.h) mit allen Konstanten des Netzes (PDF Part 1, Step 6).
        Pro Layer: ein Gewichts-Array [Neuronen][Inputs] und ein Bias-Array [Neuronen].
        """
        guard = f"{prefix}_H"
        out = [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "/* Automatisch erzeugt von nn.py - nicht von Hand ändern. */",
            "",
            f"#define {prefix}_NUM_LAYERS {len(self.layers)}",
            f"#define {prefix}_NUM_INPUTS {self.layer_sizes[0]}",
            f"#define {prefix}_NUM_OUTPUTS {self.layer_sizes[-1]}",
            "",
            f"static const int {prefix}_LAYER_SIZES[{len(self.layer_sizes)}] = "
            "{" + ", ".join(str(s) for s in self.layer_sizes) + "};",
            "",
        ]
        for l, layer in enumerate(self.layers, start=1):
            n_neurons = len(layer.neurons)
            n_inputs = len(layer.neurons[0].weights)
            out.append(f"/* Layer {l}: {n_neurons} Neuronen, je {n_inputs} Inputs */")
            out.append(f"static const float {prefix}_L{l}_WEIGHTS[{n_neurons}][{n_inputs}] = {{")
            for neuron in layer.neurons:
                out.append("    {" + ", ".join(f"{w:.8f}f" for w in neuron.weights) + "},")
            out.append("};")
            out.append(f"static const float {prefix}_L{l}_BIASES[{n_neurons}] = {{")
            out.append("    " + ", ".join(f"{n.bias:.8f}f" for n in layer.neurons))
            out.append("};")
            out.append("")
        out.append(f"#endif /* {guard} */")
        out.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out))
