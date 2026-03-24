import pandas as pd
import matplotlib.pyplot as plt
import math

# 1. Cargar el dataset unificado
try:
    df = pd.read_csv("unified_lonely_runner_topology.csv")
except FileNotFoundError:
    print(
        "Error: No se encuentra 'unified_lonely_runner_topology.csv' en el directorio."
    )
    exit()

print("Generando Gráficas de Alta Resolución para LaTeX...")

# =====================================================================
# FIGURA 1: Espectro de Medida Topológica (Los 7 Regímenes)
# =====================================================================
plt.figure(figsize=(14, 8))

# Definir estilos y etiquetas
regime_styles = {
    "Lacunary": {"color": "blue", "ls": "-", "lw": 1.5, "label": r"Lacunary ($2^i$)"},
    "Squares": {
        "color": "forestgreen",
        "ls": "-",
        "lw": 1.5,
        "label": r"Squares ($i^2$)",
    },
    "Fibonacci": {
        "color": "purple",
        "ls": "-",
        "lw": 1.5,
        "label": r"Fibonacci ($F_i$)",
    },
    "Critical": {"color": "red", "ls": "-", "lw": 1.5, "label": r"Critical ($i+k^2$)"},
    "Resonant": {
        "color": "darkorange",
        "ls": "--",
        "lw": 1.5,
        "label": r"Resonant ($12i+1$)",
    },
    "Primes": {"color": "teal", "ls": ":", "lw": 2.0, "label": r"Primes ($p_i$)"},
    "Consecutive": {
        "color": "black",
        "ls": "-",
        "lw": 2.5,
        "label": r"Consecutive ($\mu=0$)",
    },
}

for regime, style in regime_styles.items():
    subset = df[df["Regime"] == regime].sort_values("k")
    if not subset.empty:
        plt.plot(
            subset["k"],
            subset["Area_Percentage"],
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            label=style["label"],
        )

# Barreras asintóticas
plt.axhline(y=11.87, color="black", linestyle="--", alpha=0.6, linewidth=1.2)
plt.text(
    20, 11.2, r"Incompressibility Barrier $\approx 11.87\%$", color="black", fontsize=10
)

plt.axhline(y=13.53, color="black", linestyle="--", alpha=0.6, linewidth=1.2)
plt.text(
    20, 14.0, r"Kac Limit ($e^{-2}$) $\approx 13.53\%$", color="black", fontsize=10
)

# Configuración visual (Cambiado \le por \leq)
plt.title(
    r"Empirical Topological Measure Spectrum ($2 \leq k \leq 2000$)",
    fontsize=16,
    pad=15,
)
plt.xlabel(r"$k$ (Number of moving runners + 1)", fontsize=14)
plt.ylabel(r"Loneliness Area $\mu(S)$ (%)", fontsize=14)

plt.xlim(0, 2000)
plt.ylim(0, 40)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=12, loc="upper right", framealpha=0.9)

# Exportar Figura 1
plt.tight_layout()
plt.savefig("measure_spectrum_plot.pdf", format="pdf")
print("✅ Figura 1 exportada: 'measure_spectrum_plot.pdf'")
plt.close()

# =====================================================================
# FIGURA 2: Función Totiente de Euler vs Puntos Singulares
# =====================================================================
df_cons = df[df["Regime"] == "Consecutive"].sort_values("k")


def phi(n):
    amount = 0
    for k in range(1, n + 1):
        if math.gcd(n, k) == 1:
            amount += 1
    return amount


k_vals = df_cons["k"].values
empirical_points = df_cons["Singular_Points"].values
theoretical_points = [phi(int(k)) + 1 for k in k_vals]

plt.figure(figsize=(12, 6))

# Línea teórica y Puntos empíricos
plt.plot(
    k_vals,
    theoretical_points,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label=r"Theoretical $\phi(k)+1$",
    zorder=1,
)
plt.scatter(
    k_vals,
    empirical_points,
    color="black",
    s=3,
    label=r"Empirical $|S|$ (Dataset)",
    zorder=2,
)

# Configuración visual (Cambiado \le por \leq)
plt.title(
    r"Cardinality of the Survival Set $|S|$ vs. Euler's Totient $\phi(k)+1$ ($2 \leq k \leq 2000$)",
    fontsize=14,
)
plt.xlabel(r"$k$ (Number of moving runners + 1)", fontsize=12)
plt.ylabel(r"Number of Singular Points $|S|$", fontsize=12)

plt.xlim(0, 2000)
plt.ylim(0, 2000)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12, loc="upper left")

# Exportar Figura 2
plt.tight_layout()
plt.savefig("euler_totient_plot.pdf", format="pdf")
print("✅ Figura 2 exportada: 'euler_totient_plot.pdf'")
plt.close()

print("Proceso completado.")
