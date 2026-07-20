# Agent-SAW Formal Results (LaTeX-ready)

Configuration:
- seed: 42
- noise: uniform, beta=0.9, std=0.18
- selection: argmax
- detection: mean reconstructed noise (threshold=1.03)
- N=500 scenarios/task, 5 steps/trajectory

Embedded in `papaer/AnonymousSubmission2027.tex` as Table~\ref{tab:agent-saw}.

| Task | N | Acc(NW) | Acc(W) | AUC | TPR@1%FPR | Mean(W) | Mean(NW) | F1(best) | F1(rename) | F1(paraphrase) | F1(drop) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tool_selection | 500 | 1.000 | 0.736 | 0.900 | 0.632 | 1.1028 | 1.0228 | 0.817 | 0.817 | 0.770 | 0.764 |
| travel_planning | 500 | 1.000 | 0.742 | 0.892 | 0.630 | 1.1038 | 1.0290 | 0.805 | 0.805 | 0.727 | 0.760 |

```latex
\begin{table}[t]
\centering
\small
\caption{Agent-SAW results on controlled action-carrier tasks ($N{=}500$ scenarios per task; 5 steps per trajectory).}
\label{tab:agent-saw}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}l*{11}{c}@{}}
\toprule
\textbf{Task} & \textbf{N} & \textbf{Acc(NW)$\uparrow$} & \textbf{Acc(W)$\uparrow$} & \textbf{AUC$\uparrow$} & \textbf{TPR@1\%$\uparrow$} & \textbf{Mean(W)} & \textbf{Mean(NW)} & \textbf{F1(best)$\uparrow$} & \textbf{F1(ren.)$\uparrow$} & \textbf{F1(para.)$\uparrow$} & \textbf{F1(drop)$\uparrow$} \\
\midrule
Tool Selection & 500 & 1.000 & 0.736 & 0.900 & 0.632 & 1.1028 & 1.0228 & 0.817 & 0.817 & 0.770 & 0.764 \\
Travel Planning & 500 & 1.000 & 0.742 & 0.892 & 0.630 & 1.1038 & 1.0290 & 0.805 & 0.805 & 0.727 & 0.760 \\
\bottomrule
\end{tabular}%
}
\end{table}
```
