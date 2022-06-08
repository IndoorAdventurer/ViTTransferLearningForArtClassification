# Reads a csv-file showing all accuracyes and prints a latex table from it

import pandas as pd
import sys

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# print(df.columns)
# print("\n\n\n")

argmax_acc = df["accuracy_mean"].argmax()
argmax2_acc = df["accuracy_mean"].argsort()[len(df) - 2]
argmin_acc = df["accuracy_mean"].argmin()

b_argmax_acc = df["balanced_accuracy_mean"].argmax()
b_argmax2_acc = df["balanced_accuracy_mean"].argsort()[len(df) - 2]
b_argmin_acc = df["balanced_accuracy_mean"].argmin()


print(r"""\begin{table}[]
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lll}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{Balanced accuracy} \\ \hline""")

for idx, r in df.iterrows():
    model = r['model'].replace('_', '\_')
    
    acc = f"{(100 * r['accuracy_mean']):.02f}\%"
    acc_std = f"(${{\pm {(100 * r['accuracy_std']):.02f}\%}}$)"
    if idx == argmax_acc:
        acc = "\cellcolor{bestcol}" + acc
    elif idx == argmax2_acc:
        acc = "\cellcolor{secondbestcol}" + acc
    elif idx == argmin_acc:
        acc = "\cellcolor{worstcol}" + acc
    
    b_acc = f"{(100 * r['balanced_accuracy_mean']):.02f}\%"
    b_acc_std = f"(${{\pm {(100 * r['balanced_accuracy_std']):.02f}\%}}$)"
    if idx == b_argmax_acc:
        b_acc = "\cellcolor{bestcol}" + b_acc
    elif idx == b_argmax2_acc:
        b_acc = "\cellcolor{secondbestcol}" + b_acc
    elif idx == b_argmin_acc:
        b_acc = "\cellcolor{worstcol}" + b_acc
    
    print(f"\\textbf{{{model}}} & {acc} {acc_std} & {b_acc} {b_acc_std}  \\\\" + ("\hdashline" if idx == 3 else "") + ("\hline" if idx == 7 else ""))

print(r"""\end{tabular}
}
\caption{TODO}
\label{results:TODO}
\end{table}""", end="\n\n\n")