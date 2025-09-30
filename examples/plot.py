import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import sys


LQ = 0.25
UQ = 0.75
p = 1
ouralg = 'OᴘPᴀ'
figsize = (2.3, 2)

plot_kern = 'm52'
plot_crit = 'ucb'


for cc in sys.argv[1:]:
    
    if cc == 'bert8':
        cases = ['main', 'bo']
    elif cc == 'bertq8':
        cases = ['qmin']
    elif cc == 'qwen8':
        cases = ['main', 'bo']
    elif cc == 'qwenq8':
        cases = ['qmin']
    elif cc == 'llama32' or cc == 'llamab32':
        cases = ['main']
    else:
        raise ValueError
    
    for aa in cases:

        print(cc, aa)
        
        if cc == 'bert8' or cc == 'bertq8':
            case = 'bert_bz256_8gpus' + ('' if cc == 'bert8' else '_qmin')
            tmost = 20
            steps_most = 1200
            N = 20
            lb = 1.5
            ub = 2.9
            ytick = [1.6 + 0.4 * i for i in range(4)]
            seeds = list(range(10))
            s = 50
        elif cc == 'qwen8' or cc == 'qwenq8':
            case = 'qwen2_bz64_8gpus' + ('' if cc == 'qwen8' else '_qmin')
            tmost = 20
            steps_most = 600
            N = 15
            lb = 0.35
            ub = 0.45
            ytick = [0.36, 0.4, 0.44]
            seeds = list(range(10))
            s = 30
        elif cc == 'llamab32':
            case = 'llama_bz256_32gpus'
            tmost = 60
            steps_most = 300
            N = 10
            lb = 0.12
            ub = 0.215
            ytick = [0.12, 0.16, 0.2]
            seeds = list(range(10))
            s = 30
        else:
            raise ValueError

        if aa == 'main':
            algs = [
                ('random', 'Random', 'black', '-.'), 
                ('xgb', 'XGBoost', 'darkorange', '-.'),
                ('cost', 'Cost-Model*', 'magenta', ':'),
                (f'bo-m52_{plot_crit}', 'Plain-BO', 'red', '--'),
                (f'pm_et-{plot_kern}_{plot_crit}', ouralg, 'blue', '-'),
            ]
            suffix = 'main'
        elif aa == 'bo':
            algs = [
                (f'bo-m52_{plot_crit}', 'Plain BO', 'red', ':'), 
                (f'bo_et-m52_{plot_crit}', 'BO + Early Term.', 'green', '-.'),
                (f'pm-{plot_kern}_{plot_crit}', 'BO + Prior Belief', 'magenta', '--'),
                (f'pm_et-{plot_kern}_{plot_crit}', ouralg, 'blue', '-'),
            ]
            suffix = 'bo'
            lb = (ytick[0] + ytick[1]) / 2
            ytick = ytick[1:]
        elif aa == 'kern':
            algs = [
                (f'bo-m52_{plot_crit}', 'BO + M-5/2', 'red', '--'),
                (f'bo-dkm52_{plot_crit}', 'BO + DK', 'darkorange', ':'),
                (f'pm_et-m52_{plot_crit}', f'{ouralg} + M-5/2', 'purple', '-.'),
                (f'pm_et-dkm52_{plot_crit}', f'{ouralg} + DK', 'blue', '-'),
            ]
            suffix = 'kern'
        elif aa == 'crit':
            other_crit = list({'ei', 'ucb'} - {plot_crit})[0]
            algs = [
                (f'bo-{plot_kern}_{other_crit}', f'BO + {other_crit.upper()}', 'darkorange', ':'), 
                (f'bo-{plot_kern}_{plot_crit}', f'BO + {plot_crit.upper()}', 'magenta', '-.'), 
                (f'pm_et-{plot_kern}_{other_crit}', f'{ouralg} + {other_crit.upper()}', 'red', '--'),
                (f'pm_et-{plot_kern}_{plot_crit}', f'{ouralg} + {plot_crit.upper()}', 'blue', '-'),
            ]
            suffix = 'crit'
        elif aa == 'qmin':
            algs = [
                (f'qmin_2', '$q_\\text{min} = 2$', 'blue', '-'),
                (f'qmin_5', '$q_\\text{min} = 5$', 'magenta', '-'),
                (f'qmin_10', '$q_\\text{min} = 10$', 'darkorange', '--'),
                (f'qmin_20', '$q_\\text{min} = 20$', 'red', '-.'),
                (f'qmin_{s}', '$q_\\text{min} = q_\\text{max}$', 'black', ':'),
            ]
            suffix = 'qmin'
            lb = (ytick[0] + ytick[1]) / 2
            steps_most //= 2
        else:
            raise ValueError
            
        main_folder = f'./{case}/results'
        os.makedirs(f'./graphs/{case}/{suffix}', exist_ok=True)

        m = -float('inf')
        plt.figure(figsize=figsize, dpi=150)
        for a, label, color, style in algs:
            if a == 'cost':
                continue
            if 'pm' in a and '8' not in cc:
                N = N + 5
            t_regular = 1 + np.arange(50)
            y_regular_all = []
            print(a, end=' ')
            for s in seeds:
                folder = os.path.join(main_folder, f'{a}/seed_{s}/ran_queries.csv')
                if not os.path.exists(folder):
                    continue
                df = pd.read_csv(folder)
                if a == 'cost':
                    ys = df['current_best_throughput'].values
                    ys = [ys[-1] for _ in range(N)]
                else:
                    if len(df) < N:
                        continue
                    ys = df['current_best_throughput'].values
                y_regular_all.append(ys[:N])
                print(s, end=' ')
            print()
            if len(y_regular_all) > 0:
                y_regular_mean = np.median(y_regular_all, axis=0)
                y_regular_lq = np.quantile(y_regular_all, LQ, axis=0)
                y_regular_uq = np.quantile(y_regular_all, UQ, axis=0)
                plt.plot(np.arange(N)+1, y_regular_mean, style, color=color, alpha=0.8, zorder=5, label=label)
                plt.fill_between(np.arange(N)+1, y_regular_lq, y_regular_uq, alpha=0.15, zorder=2,color=color, lw=0)
                m = max(m, y_regular_uq[-1])
            if 'pm' in a and '8' not in cc:
                N = N - 5
        plt.xlabel('No. trialed PCs')
        plt.ylabel('Best $\mathcal{R}(H)$ ($s^{-1}$)')
        plt.yticks(ytick)
        plt.ticklabel_format(style='sci', axis='y', useOffset=True)
        plt.xlim((1, N if '8' in cc else N + 5))
        plt.ylim((lb, ub))
        plt.tight_layout()
        plt.savefig(f'./graphs/{case}/{suffix}/throughput-query.pdf')
        plt.close()

        print('-' * 10)

        m = - float('inf')
        plt.figure(figsize=figsize, dpi=150)
        for a, label, color, style in algs:
            print(a, end=' ')
            t_regular = np.linspace(0, tmost * 60.)
            y_regular_all = []
            for s in seeds:
                folder = os.path.join(main_folder, f'{a}/seed_{s}/ran_queries.csv')
                if not os.path.exists(folder):
                    continue
                df = pd.read_csv(folder)    
                ts = df['algtime_cumulative']
                if a == 'cost':
                    ys = df['current_best_throughput'].values
                    ys = [ys[-1] for _ in ys]
                else:
                    if len(df) < N:
                        continue
                    ys = df['current_best_throughput'].values
                ys = np.array(ys)
                indices = np.searchsorted(ts, t_regular, side="right") - 1
                indices = np.clip(indices, 0, len(ts) - 1)
                y_regular = ys[indices]
                print(s, end=' ')
                y_regular_all.append(y_regular)
            print()
            if len(y_regular_all) > 0:
                y_regular_mean = np.median(y_regular_all, axis=0)
                y_regular_lq = np.quantile(y_regular_all, LQ, axis=0)
                y_regular_uq = np.quantile(y_regular_all, UQ, axis=0)
                plt.plot(t_regular / 60., y_regular_mean, style, color=color, alpha=0.8, zorder=5, label=label)
                plt.fill_between(t_regular / 60., y_regular_lq, y_regular_uq, alpha=0.15, zorder=2,color=color, lw=0)
                m = max(m, y_regular_uq[-1])
        plt.xlabel('Time (mins)')
        plt.ylabel('Best $\mathcal{R}(H)$ ($s^{-1}$)')
        plt.yticks(ytick)
        plt.ticklabel_format(style='sci', axis='y', useOffset=True)
        plt.xlim((0, tmost))
        plt.ylim((lb, ub))
        plt.tight_layout()
        plt.savefig(f'./graphs/{case}/{suffix}/throughput-time.pdf')
        plt.close()
        
        print('-' * 10)

        m = - float('inf')
        plt.figure(figsize=figsize, dpi=150)
        for a, label, color, style in algs:
            if a == 'cost':
                continue
            print(a, end=' ')
            t_regular = np.linspace(0, steps_most)
            y_regular_all = []
            for s in seeds:
                folder = os.path.join(main_folder, f'{a}/seed_{s}/ran_queries.csv')
                if not os.path.exists(folder):
                    continue
                df = pd.read_csv(folder)    
                t_raw = df['raw_time'].map(lambda x: len(eval(x)) if 'nan' not in x else 0)
                ts = np.array(np.cumsum(t_raw))
                if a == 'cost':
                    ys = df['current_best_throughput'].values
                    ys = [ys[-1] for _ in ys]
                else:
                    if len(df) < N:
                        continue
                    ys = df['current_best_throughput'].values
                ys = np.array(ys)
                indices = np.searchsorted(ts, t_regular, side="right") - 1
                indices = np.clip(indices, 0, len(ts) - 1)
                y_regular = ys[indices]
                print(s, end=' ')
                y_regular_all.append(y_regular)
            print()
            if len(y_regular_all) > 0:
                y_regular_mean = np.median(y_regular_all, axis=0)
                y_regular_lq = np.quantile(y_regular_all, LQ, axis=0)
                y_regular_uq = np.quantile(y_regular_all, UQ, axis=0)
                plt.plot(t_regular, y_regular_mean, style, color=color, alpha=0.8, zorder=5, label=label)
                plt.fill_between(t_regular, y_regular_lq, y_regular_uq, alpha=0.15, zorder=2,color=color, lw=0)
                m = max(m, y_regular_uq[-1])
        plt.xlabel('Training steps')
        plt.ylabel('Best $\mathcal{R}(H)$ ($s^{-1}$)')
        plt.yticks(ytick)
        plt.ticklabel_format(style='sci', axis='y', useOffset=True)
        plt.xlim((0, steps_most))
        plt.ylim((lb, ub))
        plt.tight_layout()
        plt.savefig(f'./graphs/{case}/{suffix}/throughput-steps.pdf')
        plt.close()

        print('-' * 10)

        m = - float('inf')
        plt.figure(figsize=figsize, dpi=150)
        for a, label, color, style in algs:
            print(a, end=' ')
            t_regular = np.linspace(0, tmost * 60.)
            y_regular_all = []
            if a == 'cost':
                continue
            for s in seeds:
                folder = os.path.join(main_folder, f'{a}/seed_{s}/ran_queries.csv')
                if not os.path.exists(folder):
                    continue
                df = pd.read_csv(folder)  
                if len(df) < N or df['algtime_cumulative'].values[-1] < (tmost * 60.):
                    continue
                print(s, end=' ')
                y_regular_all.append(df['algtime_cumulative'][:N])
            if len(y_regular_all) > 0:
                y_regular_mean = [0] + list(np.median(y_regular_all, axis=0) / 60.)
                y_regular_lq = [0] + list(np.quantile(y_regular_all, LQ, axis=0) / 60.)
                y_regular_uq = [0] + list(np.quantile(y_regular_all, UQ, axis=0) / 60.)
                plt.plot(np.arange(N+1), y_regular_mean, style, color=color, alpha=0.8, zorder=5, label=label)
                plt.fill_between(np.arange(N+1), y_regular_lq, y_regular_uq, alpha=0.15, zorder=2,color=color, lw=0)
                m = max(m, y_regular_uq[-1])
            print()
        plt.xlim((0, N))
        plt.ylim((0, tmost + 2))
        plt.xlabel('No. trialed PCs')
        plt.ylabel('Time (mins)')
        plt.ticklabel_format(style='sci', axis='y', useOffset=True)
        plt.tight_layout()
        plt.savefig(f'./graphs/{case}/{suffix}/time-query.pdf')
        plt.close()
        
        print('-' * 10)

        T = 3 * 3600
        m = - float('inf')
        plt.figure(figsize=figsize, dpi=150)
        for a, label, color, style in algs:
            print(a, end=' ')
            if a != 'cost':
                t_regular = np.linspace(0, tmost * 60.)
                t_extrapolate = np.linspace(tmost * 60., T)
            else:
                t_extrapolate = np.linspace(0, T)
            y_regular_all = []
            y_regular_extrap = []
            for s in seeds:
                folder = os.path.join(main_folder, f'{a}/seed_{s}/ran_queries.csv')
                if not os.path.exists(folder):
                    continue
                df = pd.read_csv(folder)  
                if df['algtime_cumulative'].values[-1] < (tmost * 60.):
                    continue
                print(s, end=' ')
                ts = np.array(df['algtime_cumulative'])
                y_raw = df['raw_time'].map(lambda x: len(eval(x)) if 'nan' not in x else 0)
                y_cum = np.array(np.cumsum(y_raw))
                ys = np.interp(t_regular, ts, y_cum)
                y_regular_all.append(ys)
                best_time = np.array(df[df['algtime_cumulative'] < tmost * 60.]['current_best_time'])[-1]
                if a != 'cost':
                    y_regular_extrap.append(ys[-1] + ((t_extrapolate - t_regular[-1]) / best_time))
                else:
                    y_regular_extrap.append(t_extrapolate / best_time)
            if len(y_regular_all) > 0:
                if a != 'cost':
                    y_regular_mean = np.median(y_regular_all, axis=0)
                    y_regular_lq = np.quantile(y_regular_all, LQ, axis=0)
                    y_regular_uq = np.quantile(y_regular_all, UQ, axis=0)
                    plt.plot(y_regular_mean, t_regular / 60., style, color=color, alpha=0.8, zorder=5, label=label)
                    plt.fill_betweenx(t_regular / 60., y_regular_lq, y_regular_uq, alpha=0.15, zorder=2,color=color, lw=0)
                y_regular_mean = np.median(y_regular_extrap, axis=0)
                y_regular_lq = np.quantile(y_regular_extrap, LQ, axis=0)
                y_regular_uq = np.quantile(y_regular_extrap, UQ, axis=0)
                plt.plot(y_regular_mean, t_extrapolate / 60., style, color=color, alpha=0.8, zorder=5, label=label)
                plt.fill_betweenx(t_extrapolate / 60., y_regular_lq, y_regular_uq, alpha=0.15, zorder=2,color=color, lw=0)
                m = max(m, y_regular_uq[-1])
            print()
        plt.axhline(t_regular[-1] / 60., linestyle='--', color='grey', alpha=0.3)
        plt.xlim((0, m))
        plt.ylim((0, t_extrapolate[-1] / 60.))
        # plt.xscale('log')
        # plt.yscale('log')
        plt.ylabel('Time (mins)')
        plt.xlabel('Training steps')
        # plt.ticklabel_format(style='sci', axis='y', useOffset=True)
        plt.tight_layout()
        plt.savefig(f'./graphs/{case}/{suffix}/time-steps.pdf')
        plt.close()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for a, label, color, style in algs:
            ax.plot([1., 2.], [1., 2.], style, color=color, alpha=0.8, zorder=5, label=label)
        plt.close(fig)
        fig_leg = plt.figure(figsize=(8, 0.7), dpi=150)
        ax_leg = fig_leg.add_subplot(111)
        legend = ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=10)
        ax_leg.axis('off')
        fig_leg.canvas.draw()  # Needed to get proper bounding box
        bbox = legend.get_window_extent()
        bbox = bbox.transformed(fig_leg.dpi_scale_trans.inverted())
        fig_leg.savefig(f'./graphs/legend_{suffix}.pdf', bbox_inches=bbox, pad_inches=0)
        plt.close()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for a, label, color, style in algs:
            ax.plot([1., 2.], [1., 2.], style, color=color, alpha=0.8, zorder=5, label=label)
        plt.close(fig)
        fig_leg = plt.figure(figsize=figsize, dpi=150)
        ax_leg = fig_leg.add_subplot(111)
        legend = ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=1)
        ax_leg.axis('off')
        fig_leg.canvas.draw()  # Needed to get proper bounding box
        bbox = legend.get_window_extent()
        bbox = bbox.transformed(fig_leg.dpi_scale_trans.inverted())
        fig_leg.savefig(f'./graphs/legend_{suffix}-vertical.pdf', bbox_inches=bbox, pad_inches=0)
        plt.close()
        
        print()
    
        if suffix == 'main':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for a, label, color, style in algs:
                if a != 'cost':
                    ax.plot([1., 2.], [1., 2.], style, color=color, alpha=0.8, zorder=5, label=label)
            plt.close(fig)
            fig_leg = plt.figure(figsize=(8, 0.7), dpi=150)
            ax_leg = fig_leg.add_subplot(111)
            legend = ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=10)
            ax_leg.axis('off')
            fig_leg.canvas.draw()  # Needed to get proper bounding box
            bbox = legend.get_window_extent()
            bbox = bbox.transformed(fig_leg.dpi_scale_trans.inverted())
            fig_leg.savefig(f'./graphs/legend_{suffix}-nocost.pdf', bbox_inches=bbox, pad_inches=0)
            plt.close()
