"""Plot and export V6 CGE results; no CGE solver required."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from threadpoolctl import threadpool_limits

SECTORS = ['MATL', 'ENG', 'C27', 'TRAN', 'INFO', 'FIN', 'M72']
JOBS = 'Cumulative job-years 2027-2031'
CAPITAL = 'Capital income change in 2031'
YEARS = list(range(2027, 2032))
ANNUAL = [f'Net jobs in {y}' for y in YEARS]
STOCK = [f'{s} capital stock change in 2031 (£)' for s in SECTORS]


def plot_results(csv_file='bayesian_v6_investment_results.csv',
                 output_dir='v6_plots', show=True):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(csv_file).sort_values('Scenario')
    if results['Scenario'].duplicated().any():
        raise ValueError('Duplicate scenario IDs.')
    if 'Run signature' in results and results['Run signature'].nunique() > 1:
        raise ValueError('Mixed run signatures: use one consistent model run.')
    cols = [JOBS, CAPITAL] + SECTORS + ANNUAL + STOCK
    valid = results.loc[results['Status'].eq('Success')].copy()
    valid = valid.loc[np.isfinite(valid[cols].to_numpy(dtype=float)).all(axis=1)].copy()
    if valid.empty:
        raise ValueError('No successful finite V6 results.')
    if (valid[SECTORS] < -1e-9).any().any() or (valid[SECTORS].sum(axis=1) <= 0).any():
        raise ValueError('Invalid recipient investment allocations.')
    if not np.allclose(valid[ANNUAL].sum(axis=1), valid[JOBS], atol=1e-6, rtol=1e-8):
        raise ValueError('Cumulative job-years do not match annual employment changes.')
    print(f'Attempted: {len(results)} | Successful plotted: {len(valid)} | Omitted: {len(results)-len(valid)}')
    results.loc[~results['Scenario'].isin(valid['Scenario'])].to_csv(out/'omitted_scenarios.csv', index=False)

    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(out/f'{name}.{ext}', dpi=300, bbox_inches='tight')
        if not show:
            plt.close(fig)

    # Rankings, with the scenario ID retained in exported data.
    for col, label, name in [(JOBS, 'Cumulative net job-years (2027–2031)', 'jobs'),
                             (CAPITAL, 'Real capital income change in 2031 (model units)', 'capital')]:
        ranked = valid.sort_values(col, ascending=False).copy()
        ranked.insert(0, 'Rank', np.arange(1, len(ranked)+1))
        ranked.to_csv(out/f'{name}_rankings.csv', index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(ranked['Rank'], ranked[col], s=15, alpha=.7)
        ax.axhline(0, color='grey', linestyle=':')
        ax.set(xlabel='Allocation rank', ylabel=label, title=f'V6: {name.title()} rankings')
        ax.grid(alpha=.2)
        save(fig, f'{name}_rankings')

        history = results[['Scenario']].merge(valid[['Scenario', col]], on='Scenario', how='left')
        history['Best so far'] = history[col].cummax().ffill()
        history.to_csv(out/f'{name}_convergence.csv', index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(history['Scenario'], history['Best so far'], where='post')
        ax.set(xlabel='Scenario number', ylabel=label, title=f'V6: best {name} objective so far')
        ax.grid(alpha=.2)
        save(fig, f'{name}_convergence')

    # Non-dominated outcomes: maximise both objectives.
    ordered = valid.sort_values([JOBS, CAPITAL], ascending=False)
    frontier = ordered.loc[ordered[CAPITAL] > ordered[CAPITAL].cummax().shift(fill_value=-np.inf)].sort_values(JOBS)
    frontier.to_csv(out/'pareto_allocations.csv', index=False)
    best_jobs = valid.loc[valid[JOBS].idxmax()]
    best_capital = valid.loc[valid[CAPITAL].idxmax()]
    selected = [('Maximum employment', best_jobs), ('Maximum capital income', best_capital)]
    positive_jobs = valid.loc[valid[JOBS] > 0]
    if not positive_jobs.empty:
        selected.append(('Highest capital income with positive employment', positive_jobs.loc[positive_jobs[CAPITAL].idxmax()]))
    pd.DataFrame([{'Strategy': name, **row.to_dict()} for name, row in selected]).to_csv(out/'selected_allocations.csv', index=False)

    def tradeoff_axes(ax, title):
        ax.plot(frontier[JOBS], frontier[CAPITAL], 'k--', linewidth=1.2, label='Tested Pareto frontier')
        ax.axhline(0, color='grey', linestyle=':')
        ax.axvline(0, color='grey', linestyle=':')
        ax.set(xlabel='Cumulative net job-years (2027–2031)',
               ylabel='Real capital income change in 2031 (model units)', title=title)
        ax.grid(alpha=.2)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(valid[JOBS], valid[CAPITAL], s=22, alpha=.45, label='Successful allocations')
    tradeoff_axes(ax, 'V6: Employment–capital income trade-off')
    for name, row in selected:
        ax.scatter(row[JOBS], row[CAPITAL], s=100, marker='*', label=f'{name} (#{int(row.Scenario)})')
    ax.legend(fontsize=8)
    save(fig, 'pareto_frontier')

    # Compare 7 and 12 clusters using investment composition only.
    X = valid[SECTORS].to_numpy(dtype=float)
    X = X / X.sum(axis=1, keepdims=True)
    models, scores = {}, []
    unique = len(np.unique(X, axis=0))
    with threadpool_limits(limits=1):
        for k in range(2, min(20, len(X)-1, unique)+1):
            model = KMeans(n_clusters=k, random_state=42, n_init=30).fit(X)
            if not 2 <= len(np.unique(model.labels_)) < len(X):
                continue
            models[k] = model
            scores.append({'Clusters': k, 'Silhouette score': silhouette_score(X, model.labels_)})
    scores = pd.DataFrame(scores)
    scores.to_csv(out/'silhouette_scores.csv', index=False)
    if not scores.empty:
        best_k = int(scores.loc[scores['Silhouette score'].idxmax(), 'Clusters'])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(scores['Clusters'], scores['Silhouette score'], 'o-')
        ax.axvline(best_k, linestyle='--', label=f'Best tested k={best_k}')
        ax.set(xlabel='Number of clusters', ylabel='Mean silhouette score', title='V6: Cluster selection')
        ax.legend(); ax.grid(alpha=.2)
        save(fig, 'silhouette_scores')
    for k in (7, 12):
        if k not in models:
            print(f'Skipping k={k}: insufficient distinct allocations.')
            continue
        model = models[k]
        frame = valid.copy()
        frame['Cluster'] = model.labels_
        frame['Silhouette'] = silhouette_samples(X, model.labels_)
        order = frame.groupby('Cluster')[JOBS].mean().sort_values(ascending=False).index
        mapping = {old: i+1 for i, old in enumerate(order)}
        frame['Cluster'] = frame['Cluster'].map(mapping)
        profiles = []
        fig, ax = plt.subplots(figsize=(12, 6.5))
        for c, g in frame.groupby('Cluster'):
            shares = g[SECTORS].div(g[SECTORS].sum(axis=1), axis=0).mean()
            label = ' / '.join(f'{s} {v:.0%}' for s,v in shares.nlargest(2).items())
            ax.scatter(g[JOBS], g[CAPITAL], s=28, alpha=.75, color=plt.get_cmap('tab20')(int(c)-1),
                       label=f'{c}: {label} (n={len(g)})')
            profiles.append({'Cluster': c, 'Count': len(g), 'Profile': label,
                             'Mean job-years': g[JOBS].mean(), 'Mean capital income': g[CAPITAL].mean(),
                             'Mean silhouette': g['Silhouette'].mean(), **shares.to_dict()})
        tradeoff_axes(ax, f'V6: {k} investment-mix clusters')
        ax.legend(title='Two largest mean shares', bbox_to_anchor=(1.02,1), loc='upper left', fontsize=8)
        save(fig, f'{k}_cluster_tradeoff')
        pd.DataFrame(profiles).to_csv(out/f'{k}_cluster_profiles.csv', index=False)
        frame.to_csv(out/f'{k}_cluster_assignments.csv', index=False)

    # Annual employment and selected sector capital-stock changes.
    annual = pd.DataFrame({'Year': [2026]+YEARS})
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, row in selected:
        label = f'{name} (#{int(row.Scenario)})'
        annual[label] = [0.0] + row[ANNUAL].to_numpy(dtype=float).tolist()
        ax.plot(annual['Year'], annual[label], 'o-', label=label)
    ax.axhline(0, color='grey', linestyle=':', label='Unshocked baseline')
    ax.set(xlabel='Year', ylabel='Net additional jobs', title='V6: Annual employment impact')
    ax.set_xticks(annual['Year']); ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig, 'annual_employment')
    annual.to_csv(out/'annual_employment.csv', index=False)

    for columns, ylabel, title, filename, scale in [
            (SECTORS, 'Reallocated investment (£m)', 'Best and compromise allocations', 'selected_investment', 1),
            (STOCK, 'Capital stock change in 2031 (£m)', 'Recipient-sector capital stock effects', 'capital_stock_changes', 1e6)]:
        data = pd.DataFrame({'Sector': SECTORS})
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(SECTORS)); w = .8 / len(selected)
        for i, (name, row) in enumerate(selected):
            label = f'{name} (#{int(row.Scenario)})'
            data[label] = row[columns].to_numpy(dtype=float)/scale
            ax.bar(x+(i-(len(selected)-1)/2)*w, data[label], w, label=label)
        ax.set_xticks(x, SECTORS); ax.axhline(0, color='grey', linewidth=.8)
        ax.set(ylabel=ylabel, title=title); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
        save(fig, filename)
        data.to_csv(out/f'{filename}.csv', index=False)

    # Descriptive summary (Bayesian sampling is not uniform).
    summary = {'Attempted': len(results), 'Successful plotted': len(valid),
               'Omitted': len(results)-len(valid), 'Pearson': valid[JOBS].corr(valid[CAPITAL]),
               'Spearman': valid[JOBS].corr(valid[CAPITAL], method='spearman'),
               'Both positive': int(((valid[JOBS]>0)&(valid[CAPITAL]>0)).sum()),
               'Jobs positive, capital negative': int(((valid[JOBS]>0)&(valid[CAPITAL]<0)).sum()),
               'Jobs negative, capital positive': int(((valid[JOBS]<0)&(valid[CAPITAL]>0)).sum()),
               'Both negative': int(((valid[JOBS]<0)&(valid[CAPITAL]<0)).sum()),
               'At least one zero': int(((valid[JOBS]==0)|(valid[CAPITAL]==0)).sum()),
               'Pareto outcomes': len(frontier)}
    pd.Series(summary, name='Value').to_csv(out/'summary.csv')
    print(pd.Series(summary).to_string())
    print('\nSelected strategies:')
    for name, row in selected:
        print(f'{name}: scenario {int(row.Scenario)}, jobs={row[JOBS]:.6f}, capital={row[CAPITAL]:.6f}')
        print(row[SECTORS].to_string())
    print(f'Plots and data saved to {out.resolve()}')
    if show:
        plt.show()
    return valid, frontier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='bayesian_v6_investment_results.csv')
    parser.add_argument('--out', default='v6_plots')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    plot_results(args.csv, args.out, not args.no_show)
