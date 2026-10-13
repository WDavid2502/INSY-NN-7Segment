"""
PDF Part 2, Step 1–2: XOR trainieren, als JSON exportieren, wieder laden und vergleichen.
Architektur 2-2-1, 20000 Epochen, Lernrate 0.1.
"""

import os
import random

from nn import NeuralNetwork

SEED = 42          # fester Seed -> bei jedem Lauf dasselbe Ergebnis
EPOCHS = 20000
LEARNING_RATE = 0.1
TOLERANCE = 1e-9   # erlaubter Unterschied zwischen trainiertem und geladenem Netz

INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
TARGETS = [[0], [1], [1], [0]]


def main() -> None:
    # In den Ordner dieses Skripts wechseln, damit die Exporte immer in Code/ landen
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    random.seed(SEED)
    net = NeuralNetwork([2, 2, 1])

    print("=== Training XOR (2-2-1) ===")
    net.train(INPUTS, TARGETS, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    net.dump_json("xor_net.json")
    net.dump_txt("xor_net.txt")
    net.dump_initfile("xor_net.h", prefix="XOR")

    print("\n=== Test: trainiertes vs. geladenes Netz ===")
    loaded = NeuralNetwork.load("xor_net.json")
    all_ok = True
    for x, y in zip(INPUTS, TARGETS):
        trained_out = net.predict(x)[0]
        loaded_out = loaded.predict(x)[0]
        same = abs(trained_out - loaded_out) < TOLERANCE
        correct = round(loaded_out) == y[0]
        all_ok &= same and correct
        print(f"{x} -> trainiert {trained_out:.4f} | geladen {loaded_out:.4f} | "
              f"erwartet {y[0]} | {'OK' if correct else 'FALSCH'} | "
              f"Export/Import {'gleich' if same else 'UNTERSCHIED'}")

    print("\nErgebnis:", "alles korrekt" if all_ok else "es gibt Fehler")


if __name__ == "__main__":
    main()
