"""Model Evaluation."""
import streamlit as st
from .. import charts, ui

METRICS = [("Silhouette Score", "higher is better"), ("Davies-Bouldin Index", "lower is better"),
           ("Calinski-Harabasz Score", "higher is better")]
NICE = {"Silhouette Score": "Silhouette score", "Davies-Bouldin Index": "Davies–Bouldin index",
        "Calinski-Harabasz Score": "Calinski–Harabasz index", "Inertia": "Inertia"}


def render(ctx, go):
    ui.header("Model Evaluation", "Clustering Performance",
              "How the number of segments was chosen and how stable the K-Means result is.")
    e5 = ctx.eval5
    if e5 is None:
        ui.unavailable("The K-Means evaluation results")
        return
    best = {m: int(e5.loc[e5[m].idxmin() if "lower" in b else e5[m].idxmax(), "K"]) for m, b in METRICS}
    chosen = best["Silhouette Score"]
    row = e5.set_index("K").loc[chosen]
    ui.kpis([("Selected K", f"{chosen}", "number of customer segments"),
             ("Silhouette score", f"{row['Silhouette Score']:.4f}", "range −1 to 1 · higher is better"),
             ("Davies–Bouldin index", f"{row['Davies-Bouldin Index']:.4f}", "0 or more · lower is better"),
             ("Calinski–Harabasz index", f"{row['Calinski-Harabasz Score']:,.2f}", "higher is better")])
    agree = len(set(best.values())) == 1
    ui.callout(f"K = {chosen} is the best of the tested values (K = 2 to 8) on the silhouette score, the Davies–Bouldin index and the "
               f"Calinski–Harabasz index{' — all three agree' if agree else ''}.", "calculated")
    ui.callout("Internal clustering metrics describe how compact and well separated the clusters are. They do not prove that the "
               "segments are objectively correct or commercially useful.", "warn", label="Keep in mind")

    t1, t3 = st.tabs(["Selecting K", "Understanding the metrics"])
    with t1:
        cols = st.columns(2, gap="large")
        for i, (m, b) in enumerate([("Inertia", "lower is better · look for the elbow")] + METRICS):
            with cols[i % 2]:
                charts.show(charts.metric_curve(e5[["K", m]].rename(columns={m: NICE[m]}), NICE[m], NICE[m], chosen, b), key=f"ev_{i}")
        st.markdown("<div class='small'>The dashed line marks the selected K.</div>", unsafe_allow_html=True)
        with st.expander("Detailed results for K = 2 to 8"):
            st.dataframe(e5.rename(columns={"Silhouette Score": "Silhouette", "Davies-Bouldin Index": "Davies–Bouldin",
                                            "Calinski-Harabasz Score": "Calinski–Harabasz", "Smallest Cluster": "Smallest", "Largest Cluster": "Largest"}),
                         hide_index=True, **ui.FULL, column_config={
                             "Inertia": st.column_config.NumberColumn(format="%.2f"),
                             "Silhouette": st.column_config.NumberColumn(format="%.4f"),
                             "Davies–Bouldin": st.column_config.NumberColumn(format="%.4f"),
                             "Calinski–Harabasz": st.column_config.NumberColumn(format="%.2f")})
    with t3:
        st.markdown(
            "| Metric | What it measures | Better when |\n|---|---|---|\n"
            "| Inertia | Total squared distance of customers to their segment centre | Lower, but it always falls as K grows — look for the ‘elbow’ |\n"
            "| Silhouette score | How much closer each customer is to its own segment than to the next one (−1 to 1) | Higher |\n"
            "| Davies–Bouldin index | Average similarity between each segment and its most similar neighbour | Lower |\n"
            "| Calinski–Harabasz index | Ratio of between-segment to within-segment spread | Higher |\n")
        st.markdown(
            "**Why not accuracy?** Classification accuracy needs true labels to compare against. Customer segments are discovered from "
            "the data, so no true labels exist and a train/test split does not apply.\n\n"
            "**Stability.** Refitting K = 3 with five other random seeds produced practically the same segments (Adjusted Rand Index ≈ 0.999).")
        with st.expander("Model configuration"):
            st.markdown("- Algorithm: K-Means (scikit-learn), `n_init = 10`, `random_state = 42`\n"
                        "- Features: Recency, log(1 + Frequency), log(1 + Monetary), each standardised\n"
                        "- Values of K tested: 2 to 8\n- Stability: K = 3 refitted with five other random seeds")
