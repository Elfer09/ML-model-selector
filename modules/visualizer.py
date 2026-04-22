"""
visualizer.py
Auto-generates interactive Plotly charts based on user column selection.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


CHART_TYPES = [
    "Histogram",
    "Box Plot",
    "Scatter Plot",
    "Correlation Heatmap",
    "Bar Chart",
    "Line Chart",
]


def auto_charts(df: pd.DataFrame):
    """Render chart builder UI and generate charts."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found. Charts require numeric data.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        chart_type = st.selectbox("Chart Type", CHART_TYPES)

        if chart_type == "Histogram":
            x_col = st.selectbox("Column", numeric_cols, key="hist_x")
            bins = st.slider("Bins", 5, 100, 30)
            color_col = st.selectbox("Color by (optional)", ["None"] + all_cols, key="hist_color")

        elif chart_type == "Box Plot":
            y_col = st.selectbox("Value Column", numeric_cols, key="box_y")
            x_col = st.selectbox("Group by (optional)", ["None"] + all_cols, key="box_x")

        elif chart_type == "Scatter Plot":
            x_col = st.selectbox("X Axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y Axis", numeric_cols, key="scatter_y")
            color_col = st.selectbox("Color by (optional)", ["None"] + all_cols, key="scatter_color")
            trendline = st.checkbox("Add trendline", value=True)

        elif chart_type == "Correlation Heatmap":
            selected_cols = st.multiselect(
                "Columns to include",
                numeric_cols,
                default=numeric_cols[:min(8, len(numeric_cols))],
            )

        elif chart_type == "Bar Chart":
            x_col = st.selectbox("X Axis (category)", all_cols, key="bar_x")
            y_col = st.selectbox("Y Axis (value)", numeric_cols, key="bar_y")
            agg_func = st.selectbox("Aggregation", ["mean", "sum", "count", "median"])

        elif chart_type == "Line Chart":
            x_col = st.selectbox("X Axis", all_cols, key="line_x")
            y_col = st.selectbox("Y Axis", numeric_cols, key="line_y")
            color_col = st.selectbox("Color by (optional)", ["None"] + all_cols, key="line_color")

    with col2:
        try:
            if chart_type == "Histogram":
                color = color_col if color_col != "None" else None
                fig = px.histogram(df, x=x_col, nbins=bins, color=color,
                                   title=f"Distribution of {x_col}",
                                   template="plotly_dark")

            elif chart_type == "Box Plot":
                x = x_col if x_col != "None" else None
                fig = px.box(df, x=x, y=y_col,
                             title=f"Box Plot: {y_col}" + (f" by {x_col}" if x else ""),
                             template="plotly_dark")

            elif chart_type == "Scatter Plot":
                color = color_col if color_col != "None" else None
                trend = "ols" if trendline else None
                fig = px.scatter(df, x=x_col, y=y_col, color=color,
                                 trendline=trend,
                                 title=f"{x_col} vs {y_col}",
                                 template="plotly_dark",
                                 opacity=0.7)

            elif chart_type == "Correlation Heatmap":
                if len(selected_cols) < 2:
                    st.warning("Select at least 2 columns for a heatmap.")
                    return
                corr = df[selected_cols].corr()
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        colorscale="RdBu_r",
                        zmin=-1, zmax=1,
                        text=corr.round(2).values,
                        texttemplate="%{text}",
                    )
                )
                fig.update_layout(title="Correlation Heatmap", template="plotly_dark")

            elif chart_type == "Bar Chart":
                grouped = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                grouped = grouped.sort_values(y_col, ascending=False).head(20)
                fig = px.bar(grouped, x=x_col, y=y_col,
                             title=f"{agg_func.title()} of {y_col} by {x_col} (top 20)",
                             template="plotly_dark")

            elif chart_type == "Line Chart":
                color = color_col if color_col != "None" else None
                fig = px.line(df, x=x_col, y=y_col, color=color,
                              title=f"{y_col} over {x_col}",
                              template="plotly_dark")

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Chart error: {e}. Try different columns or chart type.")