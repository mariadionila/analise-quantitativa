import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page: #f6f8fb;
            --paper: #ffffff;
            --paper-soft: #edf2f7;
            --ink: #14213d;
            --muted: #637083;
            --line: #d8e0ea;
            --sapphire: #255f85;
            --coral: #ef6f61;
            --amber: #f2b84b;
            --green: #3c8d7a;
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(37, 95, 133, 0.08), transparent 18rem),
                var(--page);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.4rem;
        }

        h1, h2, h3, p, span, label, div {
            color: var(--ink);
            letter-spacing: 0;
        }

        h1 {
            font-size: 2.2rem;
            line-height: 1.08;
            font-weight: 780;
            margin-bottom: 0.35rem;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }

        .hero {
            background: var(--paper);
            border: 1px solid var(--line);
            border-left: 7px solid var(--coral);
            border-radius: 8px;
            padding: 1.45rem 1.6rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 18px 42px rgba(20, 33, 61, 0.08);
        }

        .eyebrow {
            color: var(--coral);
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.45rem;
        }

        .hero p {
            color: var(--muted);
            font-size: 1rem;
            max-width: 820px;
            margin-bottom: 0;
        }

        .card {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.1rem 1.15rem;
            min-height: 164px;
            box-shadow: 0 12px 28px rgba(20, 33, 61, 0.07);
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: linear-gradient(180deg, var(--sapphire), var(--coral));
        }

        .card h3 {
            margin-top: 0.72rem;
            margin-bottom: 0.4rem;
            font-size: 1.08rem;
            color: var(--ink);
        }

        .card p, .card li {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .badge {
            display: inline-flex;
            padding: 0.22rem 0.54rem;
            border-radius: 6px;
            background: rgba(37, 95, 133, 0.1);
            color: var(--sapphire);
            border: 1px solid rgba(37, 95, 133, 0.16);
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        [data-testid="metric-container"] {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 26px rgba(20, 33, 61, 0.06);
        }

        [data-testid="metric-container"] label,
        [data-testid="metric-container"] [data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--sapphire);
        }

        .stButton > button {
            background: var(--sapphire);
            color: white;
            border: 0;
            border-radius: 8px;
            font-weight: 760;
            padding: 0.55rem 1rem;
            box-shadow: 0 10px 20px rgba(37, 95, 133, 0.18);
        }

        .stButton > button:hover {
            background: #1d4f70;
            color: white;
            border: 0;
        }

        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        div[data-baseweb="select"] > div,
        textarea {
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
            border-radius: 8px !important;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--line);
        }

        .stAlert {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, eyebrow: str = "Analise Quantitativa") -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_axis(fig, ax) -> None:
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#637083")
    ax.yaxis.label.set_color("#14213d")
    ax.xaxis.label.set_color("#14213d")
    ax.title.set_color("#14213d")
    for spine in ax.spines.values():
        spine.set_color("#d8e0ea")
