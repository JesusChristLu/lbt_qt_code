# -*- coding: utf-8 -*-
# @Time     : 2022/9/21 16:56
# @Author   : WTL
# @Software : PyCharm
import copy
from typing import Union
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import qutip as qp
from matplotlib import cm
from pathlib import Path
from datetime import datetime
from collections.abc import Iterable
from mpl_toolkits.mplot3d import Axes3D
import scienceplots
plt.style.use(['science', 'ieee'])

from pulse.pulse_base import XYPulseBase, ZPulseBase
from pulse.pulse_lib import Constant
from functions import *

CMAP_LIST = [
    'cividis',
    'viridis',
    'plasma',
    'inferno',
    'magma',
    'spring',
    'summer',
    'autumn',
    'winter',
    'cool',
    'copper',
    'Blues',
    'Greens',
    'gist_rainbow',
    'rainbow',
    'jet',
    'turbo',
    'coolwarm',
    'RdBu'
]


class PlotTool:
    def __init__(self, **plot_params):
        """
        创建一个绘图类，通过修改rcParams统一配置绘图属性
        :param plot_params: 绘图参数
        """
        if plot_params is None:
            plot_params = {}
        self.flag_save = plot_params.get('flag_save', False)
        self.flag_close = plot_params.get('flag_close', False)
        self.args = {}
        self.root_path = Path(plot_params.get('root_path', Path.cwd() / 'figures'))

        self.cmap = cm.get_cmap(plot_params.get('cmap', CMAP_LIST[-1]))
        self.alpha = plot_params.get('alpha', 0.7)
        # self.plot_style = plot_params.get('plot_style', 'bmh')
        self.units = plot_params.get(
            'units',
            {
                'x-t': 'ns',
                'x-w': 'GHz',
                'x-vol': 'V',
                'y-t': 'ns',
                'y-w': 'MHz',
                'y-vol': 'V',
            },
        )

        self.xscale = plot_params.get('xscale', 'linear')
        self.yscale = plot_params.get('yscale', 'linear')
        self.zscale = plot_params.get('zscale', 'linear')

        # plt.style.use(self.plot_style)
        self.custom_rc = {
            'figure.figsize': plot_params.get('figsize', (6, 4)),
            'figure.dpi': plot_params.get('figure_dpi', 120),
            'savefig.format': plot_params.get('savefig_format', 'png'),
            'savefig.dpi': plot_params.get('savefig_dpi', 300),
            # 'image.cmap': plot_params.get('cmap', CMAP_LIST[-1]),
            'font.size': plot_params.get('font_size', 11),
            'axes.titlesize': plot_params.get('axes_titlesize', 16),
            'axes.labelsize': plot_params.get('axes_labelsize', 15),
            'xtick.labelsize': plot_params.get('xtick_labelsize', 15),
            'ytick.labelsize': plot_params.get('ytick_labelsize', 15),
            'lines.linewidth': plot_params.get('lines_linewidth', 4),
            'lines.marker': plot_params.get('lines_marker', ''),
            'lines.markersize': plot_params.get('lines_markersize', 8),
            'legend.fontsize': plot_params.get('legend_fontsize', 12),
        }
        plt.rcParams.update(self.custom_rc)
        self.default_attrs = copy.deepcopy(self.__dict__)

    def reset(self):
        self.default_attrs.update({'args': self.args})
        self.__init__(**self.default_attrs)

    def save_fig(
            self,
            fig,
            save_name: str,
            flag_save: bool = None
    ):
        flag_save = self.flag_save if flag_save is None else flag_save
        if flag_save:
            plt.tight_layout()

            date = datetime.now().strftime('%Y-%m')
            time = datetime.now().strftime('%m%d-%H.%M.%S')

            current_path = self.root_path / date
            current_path.mkdir(parents=True, exist_ok=True)
            save_path = (
                current_path / f'{time}_{save_name}.{self.custom_rc["savefig.format"]}'
            )
            print(f'fig path: {save_path}')
            fig.savefig(save_path)

            if self.flag_close:
                plt.close(fig)
                # plt.close('all')

    def unify_units(
        self,
        axis: str,
        unit_type: str,
        label: str,
        values: Union[np.ndarray, tuple, dict],
    ):
        unit_key = f'{axis}-{unit_type}'
        unit_suffix = self.units.get(unit_key)

        if unit_suffix is None:
            return label, values
        elif unit_suffix == r'$\mu$s':
            unit_coef = 1e-3
        elif unit_suffix == 'MHz':
            unit_coef = 1e3
        elif unit_suffix == 'mV':
            unit_coef = 1e3
        else:
            unit_coef = 1

        label += f'({unit_suffix})'
        if isinstance(values, np.ndarray):
            values *= unit_coef
        elif isinstance(values, tuple):
            values = (val * unit_coef for val in values)
        elif isinstance(values, dict):
            values = {key: val * unit_coef for key, val in values.items()}

        return label, values

    def plot(
            self,
            x,
            y: Union[list[np.ndarray], np.ndarray],
            y_name: Union[list, str],
            *args,
            fig=None, ax=None, xtype=None, ytype=None,
            xlabel='', ylabel='', title='',
            xlim=None, ylim=None,
            flag_legend: bool = True, loc: str = None,
            tick_param: dict = None, rotation: float = None,
            save_name=None, flag_save=None,
            **kwargs
    ):
        x = np.asarray(x)
        y2d = np.atleast_2d(y)
        y_name = np.atleast_1d(y_name)
        if xtype in ['t', 'w']:
            xlabel, x = self.unify_units('x', xtype, xlabel, x)
        if ytype in ['t', 'w']:
            ylabel, y2d = self.unify_units('y', ytype, ylabel, y2d)

        tick_param = {'num_x': 10, 'digits_x': 2} if tick_param is None else tick_param
        x_pos, xticklabels = normalize_tick_label('x', x, tick_param)

        if fig is None:
            fig, ax = plt.subplots()

        if rotation is None:
            if not isinstance(x[0], Iterable):
                rotation = 0
            elif len(x[0]) == 1:
                rotation = 0
            else:
                rotation = -60

        custom_color = kwargs.get('color', self.cmap(np.linspace(0, 1, len(y2d))))

        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(xticklabels, rotation=rotation)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

        plot_kwargs = copy.deepcopy(kwargs)
        plot_kwargs.pop('loc', None)
        plot_kwargs.pop('color', None)

        self.args.update({
            ax: [],
            'label': y_name
        })
        for i, name in enumerate(y_name):
            name = fr'{name}'
            (line,) = ax.plot(
                np.arange(len(x)),
                y2d[i],
                *args,
                label=name,
                color=custom_color[i],
                alpha=self.alpha,
                **plot_kwargs,
            )
            self.args[ax].append(line)

        if len(self.args[ax]) == 1:
            self.args[ax] = self.args[ax][0]
            self.args['label'] = self.args['label'][0]

        if flag_legend:
            ax.legend(loc=loc)

        if save_name is None:
            save_name = title
        self.save_fig(fig, save_name, flag_save)

    def plot_pulse(
        self,
        pulse: Union[XYPulseBase, ZPulseBase],
        data_type: str = 'w',
        xlabel: str = r'time',
        ylabel: str = None,
        title: str = None,
        plot_type: str = 'pulse',
        save_name=None,
        flag_psd: bool = False,
        offset_psd: float = None,
        cut_off_psd: float = None,
    ):
        t = np.array(pulse.t)
        xlabel, t = self.unify_units('x', 't', xlabel, t)
        data = np.array(pulse.data) / 2 / np.pi

        if data_type == 'w':
            if ylabel is None and isinstance(pulse, XYPulseBase):
                ylabel = r'$\Omega$'
            if ylabel is None and isinstance(pulse, ZPulseBase):
                ylabel = r'$\omega$'

            ylabel, data = self.unify_units('y', 'w', ylabel, data)
            try:
                envelope = {
                    key: value / 2 / np.pi for key, value in pulse.envelope.items()
                }
                _, envelope = self.unify_units('y', 'w', '', envelope)
            except (Exception,):
                pass
        elif data_type == 'flux':
            ylabel = 'Voltage' if ylabel is None else ylabel
            data = freq2flux(data, **pulse.q_dic)

            ylabel, data = self.unify_units('y', 'vol', ylabel, data)
            try:
                envelope = {
                    key: freq2flux(value / 2 / np.pi, **pulse.q_dic)
                    for key, value in pulse.envelope.items()
                }
                _, envelope = self.unify_units('y', 'vol', '', envelope)
            except (Exception,):
                pass
        elif data_type == 'g':
            ylabel = 'g' if ylabel is None else ylabel
            rho_pair, *_ = pulse.rho_map.keys()
            rho_value, *_ = pulse.rho_map.values()
            ql, c, qr = rho_pair.split('-')
            wl, wr = [pulse.q_dic[bit]['w'] for bit in (ql, qr)]
            data = wc2geff(data, wl, wr, rho_value)

            ylabel, data = self.unify_units('y', 'w', ylabel, data)
        else:
            raise ValueError(f'data_type {data_type} is not supported.')

        line_width = 0.8 * self.custom_rc['lines.linewidth']
        marker_size = 0.6 * self.custom_rc['lines.markersize']

        fig = plt.figure()
        if plot_type == 'pulse':
            ax_num = 1 + flag_psd
            ax = fig.add_subplot(ax_num, 1, 1)
            ax.plot(t, data, linewidth=line_width, markersize=marker_size)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title if title else 'Pulse')
        elif plot_type == 'envelope':
            ax_num = 1 + flag_psd
            ax = fig.add_subplot(ax_num, 1, 1)
            ax.plot(
                t,
                envelope['X'],
                label='X',
                linewidth=line_width,
                markersize=marker_size,
            )
            ax.plot(
                t,
                envelope['Y'],
                label='Y',
                linewidth=line_width,
                markersize=marker_size,
            )
            ax.legend()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title if title else 'Envelope')
        elif plot_type == 'both':
            ax_num = 2 + flag_psd
            ax1 = fig.add_subplot(ax_num, 1, 1)
            ax1.plot(t, data, 'o-', linewidth=line_width, markersize=marker_size)
            ax1.set_ylabel(ylabel)
            ax1.set_title(title[0] if title else 'Pulse')

            ax2 = fig.add_subplot(ax_num, 1, 2, sharex=ax1, sharey=ax1)
            ax2.plot(
                t,
                envelope['X'],
                'o-',
                label='X',
                linewidth=line_width,
                markersize=marker_size,
            )
            ax2.plot(
                t,
                envelope['Y'],
                'o-',
                label='Y',
                linewidth=line_width,
                markersize=marker_size,
            )
            ax2.legend()
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel(ylabel)
            ax2.set_title(title[1] if title else 'Envelope')
        else:
            raise ValueError(f'plot_type {plot_type} is not supported.')

        if flag_psd:
            if offset_psd is None:
                offset_psd = 5 * pulse.width
            offset_pulse = Constant(width=offset_psd, arg=0)
            offset_pulse.get_pulse()
            pulse_all = copy.deepcopy(offset_pulse) + pulse + offset_pulse
            pulse_list = pulse_all.data / 2 / np.pi

            (
                basic_freq,
                basic_amp,
                basic_phase,
                basic_offset,
                peaks,
                peaks_fwhm,
                df,
                freq_pulse,
                psd_pulse,
            ) = pulse_psd(pulse_list, pulse.sample_rate)
            peaks_width, peaks_width_height, peaks_left, peaks_right = peaks_fwhm
            peaks_width = peaks_width * df

            if cut_off_psd is None:
                cut_off_psd = 10 * peaks_width

            ax_psd = fig.add_subplot(ax_num, 1, ax_num)
            ax_psd.set_title(
                fr'$\delta f$={round(df * 1e3, 3)},$f_0$={np.around(basic_freq * 1e3, 3)},'
                fr'$f_{{FWHM}}$={np.around(peaks_width * 1e3, 3)}MHz'
            )
            ax_psd.set_xlim(
                max([0, basic_freq - cut_off_psd]), basic_freq + cut_off_psd
            )
            ax_psd.stem(freq_pulse, psd_pulse)
            ax_psd.hlines(
                peaks_width_height,
                peaks_left * df,
                peaks_right * df,
                color='tab:grey',
                label='FWHM',
            )
            ax_psd.set_xlabel('freq(GHz)')
            ax_psd.set_ylabel('PSD(W/Hz)')

        if save_name is None:
            save_name = title if title else 'Pulse'
        self.save_fig(fig, save_name)

    def plot_bloch(
        self,
        states=None,
        points=None,
        view_angle=(-60, 30),
        fig=None,
        save_name: str = None,
        flag_save=None
    ):
        if fig is None:
            fig = plt.figure()

        b = qp.Bloch(fig=fig, view=view_angle)
        b.view = view_angle
        if states is not None:
            custom_color = self.cmap(np.linspace(0, 1, len(states)))
            b.vector_color = custom_color
            b.vector_width = 2
            b.add_states(states)
            b.render()
        if points is not None:
            custom_color = self.cmap(np.linspace(0, 1, len(points)))
            b.point_color = custom_color
            b.add_points(points, 'm')
            b.render()

        if save_name is None:
            save_name = 'Bloch view'
        plt.show()
        self.save_fig(fig, save_name, flag_save)

    def plot_fft(
        self,
        x,
        y,
        ytype='w',
        xlabel='FFT freq',
        ylabel='|FFT|',
        title='FFT Spectrum',
        fft_freq_bound=np.inf,
        fig=None,
        ax=None,
        save_name=None,
    ):
        freq0, amp0, phase0, offset0, peaks, xfft, yfft = fit_fft(
            np.array(x), np.array(y), freq_max=fft_freq_bound
        )
        if ytype == 'w':
            amp0 /= 2 * np.pi
            yfft /= 2 * np.pi

        xlabel, xfft = self.unify_units('x', 'w', xlabel, xfft)
        ylabel, (yfft, amp0, offset0) = self.unify_units(
            'y', ytype, ylabel, (yfft, amp0, offset0)
        )
        x_w = self.units['x-w']

        if fig is None:
            fig, ax = plt.subplots()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            title
            + f'(peak freq: {freq0:.3f}{x_w})\namp: {amp0:.3f}, phase: {phase0:.3f}, '
            f'offset={offset0:.3f}'
        )
        marker_size = self.custom_rc['lines.markersize']
        line_width = self.custom_rc['lines.linewidth']
        self.custom_rc['lines.markersize'] *= 0.8
        self.custom_rc['lines.linewidth'] *= 0.8
        plt.rcParams.update(self.custom_rc)
        ax.stem(xfft, yfft, use_line_collection=True, linefmt='grey')
        ax.stem(xfft[peaks], yfft[peaks], linefmt='grey', markerfmt='C1X')
        self.custom_rc['lines.markersize'] = marker_size
        self.custom_rc['lines.linewidth'] = line_width

        if save_name is None:
            save_name = title
        self.save_fig(fig, save_name)

    def plot_levels(
        self,
        x,
        energy_dress: dict,
        energy_bare: dict,
        excit_total: Union[int, tuple] = None,
        excit_1q: Union[int, tuple] = None,
        xtype='w',
        xlabel=r'$\omega$',
        ylabel='Energies',
        title='Qubit Level Spectrum',
        cmap='gist_rainbow',
        label_pos: int = -1,
        figsize=(7, 14),
        tick_param: dict = None,
        rotation: float = None,
        save_name: str = None,
    ):
        """
        传入缀饰态和裸态能量，绘制能级图
        :param x: 比特频率、zamp或其他变量
        :param energy_dress: 缀饰态能量，结构为{(0,...,0): array([...]), ..., (1,...,1): array([...])}
        :param energy_bare: 裸态能量，结构同上
        :param excit_total: 允许的总激发子个数，格式为(lb, ub)或ub[表示(0, ub)]
        :param excit_1q: 单比特上允许存在的激发子个数，格式同上
        :param xtype:
        :param xlabel:
        :param ylabel:
        :param title:
        :param cmap:
        :param label_pos: 标签的位置，值为索引
        :param figsize:
        :param tick_param:
        :param rotation:
        :param save_name:
        :return:
        """
        x = np.asarray(x)
        xlabel, x = self.unify_units('x', xtype, xlabel, x)
        ylabel, energy_dress = self.unify_units('y', 'w', ylabel, energy_dress)
        _, energy_bare = self.unify_units('y', 'w', ylabel, energy_bare)

        excit_total = np.inf if excit_total is None else excit_total
        excit_1q = np.inf if excit_1q is None else excit_1q
        lb_total = 0 if not isinstance(excit_total, tuple) else excit_total[0]
        ub_total = (
            excit_total if not isinstance(excit_total, tuple) else excit_total[-1]
        )
        lb_1q = 0 if not isinstance(excit_1q, tuple) else excit_1q[0]
        ub_1q = excit_1q if not isinstance(excit_1q, tuple) else excit_1q[-1]

        bare_states = []
        dress_states = []
        for i, bare_dress in enumerate(
            zip(list(energy_bare.keys()), list(energy_dress.keys()))
        ):
            bare, dress = bare_dress
            if not (lb_total <= sum(bare) <= ub_total):
                continue
            if not (lb_1q <= max(bare) <= ub_1q):
                continue
            bare_states.append(bare)
            dress_states.append(dress)

        tick_param = {'num_x': 10, 'digits_x': 2} if tick_param is None else tick_param
        x_pos, xticklabels = normalize_tick_label('x', x, tick_param)
        if rotation is None:
            if not isinstance(x[0], Iterable):
                rotation = 0
            elif len(x[0]) == 1:
                rotation = 0
            else:
                rotation = -60

        fig, ax = plt.subplots(figsize=figsize)
        cmap = cm.get_cmap(cmap)
        custom_color = cmap(np.linspace(0, 1, len(bare_states)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(xticklabels, rotation=rotation)
        xidx = np.arange(len(x))
        for i, bare_dress in enumerate(zip(bare_states, dress_states)):
            bare, dress = bare_dress
            if not (lb_total <= sum(bare) <= ub_total):
                continue
            if not (lb_1q <= max(bare) <= ub_1q):
                continue

            label = fr'$\left|{"".join([str(s) for s in bare])}\right>$'
            ax.plot(
                xidx,
                energy_dress[dress],
                '.',
                color=custom_color[i],
                alpha=0.8 * self.alpha,
            )

            line_width = 0.8 * self.custom_rc['lines.linewidth']
            ax.plot(
                xidx,
                energy_bare[bare],
                ':',
                color=custom_color[i],
                alpha=self.alpha,
                linewidth=line_width,
            )
            ax.annotate(label, (xidx[label_pos], energy_bare[bare][label_pos]))

        if save_name is None:
            save_name = title
        self.save_fig(fig, save_name)

    def plot_coefs(
        self,
        x,
        coef_dic: dict,
        xlabel=r'$\Omega$',
        ylabel='Coefficients',
        title='Pauli Coefficients',
        save_name: str = None,
    ):
        x = np.asarray(x)

        if xlabel == r'$\Omega$':
            xlabel, x = self.unify_units('x', 'w', xlabel, x)
        ylabel, coef_dic = self.unify_units('y', 'w', ylabel, coef_dic)

        fig, ax = plt.subplots()
        custom_color = self.cmap(np.linspace(0, 1, len(coef_dic)))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for i, (key, y) in enumerate(coef_dic.items()):
            ax.plot(x, y, label=key, color=custom_color[i], alpha=self.alpha)

        ax.legend()

        if save_name is None:
            save_name = title
        self.save_fig(fig, save_name)

    def plot_lines(
        self,
        x,
        y_list, y_name_list,
        xtype='t', xlabel=r't',
        ytype='', ylabel=r'Expectation',
        title='Qubit State Evolution',
        marker='o',
        cmap=None,
        save_name: str = None,
        xlim=None, ylim=None,
        tick_param: dict = None, rotation: float = None,
        fig=None, ax=None,
        flag_save=None,
        **kwargs
    ):
        if marker:
            plt.rcParams.update({'lines.marker': marker})
        if cmap:
            self.cmap = cm.get_cmap(cmap)

        x = np.asarray(x)
        y_list = np.asarray(y_list)

        xlabel, x = self.unify_units('x', xtype, xlabel, x)
        ylabel, y_list = self.unify_units('y', ytype, ylabel, y_list)

        tick_param = {'num_x': 5, 'digits_x': 1} if tick_param is None else tick_param
        x_pos, xticklabels = normalize_tick_label('x', x, tick_param)

        if fig is None:
            fig, ax = plt.subplots()

        if rotation is None:
            if not isinstance(x[0], Iterable):
                rotation = 0
            elif len(x[0]) == 1:
                rotation = 0
            else:
                rotation = -60

        custom_color = kwargs.get(
            'color', self.cmap(np.linspace(0, 1, len(y_name_list)))
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(xticklabels, rotation=rotation)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

        plot_kwargs = copy.deepcopy(kwargs)
        plot_kwargs.pop('loc', None)
        plot_kwargs.pop('color', None)
        for i, y_name in enumerate(y_name_list):
            label = fr'{y_name}'
            ax.plot(
                np.arange(len(x)),
                y_list[i],
                label=label,
                color=custom_color[i],
                alpha=self.alpha,
                **plot_kwargs,
            )

        ax.legend(loc=kwargs.get('loc'))

        if save_name is None:
            save_name = title
        self.save_fig(fig, save_name, flag_save)

    def plot_lines_fit(
        self,
        x,
        y_list, y_name_list,
        fit_func_list, fit_name_list=None,
        xtype='t', xlabel=r't',
        ytype='', ylabel=r'Expectation',
        title='Qubit State Evolution',
        marker='o', save_name: str = None,
        fig=None, ax=None,
        flag_save=None,
        **kwargs
    ):
        x = np.asarray(x)
        x_interp = np.linspace(x[0], x[-1], 100 * len(x))
        y_list = np.asarray(y_list)

        xlabel, (x, x_interp1) = self.unify_units('x', xtype, xlabel, (x, x_interp))
        ylabel, y_list = self.unify_units('y', ytype, ylabel, y_list)

        if fig is None:
            fig, ax = plt.subplots()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        custom_color_data = self.cmap(np.linspace(0, 1, len(y_name_list)))
        for i, y_name in enumerate(y_name_list):
            label = fr'{y_name}'
            ax.scatter(
                x,
                y_list[i],
                label=label,
                color=custom_color_data[i],
                marker=marker,
                alpha=self.alpha,
                **kwargs,
            )
            ax.legend()

        if fit_name_list is None:
            fit_name_list = [expec_name + '_fit' for expec_name in y_name_list]
        custom_color_fit = self.cmap(np.linspace(0, 1, len(fit_name_list)))
        for i, fit_name in enumerate(fit_name_list):
            label = fr'{fit_name}'
            _, y_fit = self.unify_units('y', ytype, ylabel, fit_func_list[i](x_interp))
            ax.plot(
                x_interp1,
                y_fit,
                '-',
                label=label,
                color=custom_color_fit[i],
                alpha=self.alpha,
                **kwargs,
            )
            ax.legend()

        if save_name is None:
            save_name = title
        self.save_fig(fig, save_name, flag_save)

    def plot_heatmap(
        self,
        x,
        y,
        Z,
        xlabel=r'$\omega$',
        ylabel=r't',
        zlabel=r'',
        xtype='w',
        ytype='t',
        title='',
        flag_unify_x: bool = True,
        flag_unify_y: bool = True,
        save_name: str = None,
        fig=None,
        ax=None,
        rotation: float = None,
        units: dict = None,
        tick_param: dict = None,
        norm: str = None,
    ):
        x = np.array(x)
        y = np.array(y)
        Z = np.array(Z)
        if units:
            self.units.update(units)
        if flag_unify_x:
            xlabel, x = self.unify_units('x', xtype, xlabel, x)
        if flag_unify_y:
            ylabel, y = self.unify_units('y', ytype, ylabel, y)

        X, Y = np.meshgrid(np.arange(len(x)), np.arange(len(y)))

        tick_param = (
            {'num_x': 10, 'digits_x': 2, 'num_y': 10, 'digits_y': 3}
            if tick_param is None
            else tick_param
        )
        x_pos, xticklabels = normalize_tick_label('x', x, tick_param)
        y_pos, yticklabels = normalize_tick_label('y', y, tick_param)

        if fig is None:
            fig, ax = plt.subplots()

        if rotation is None:
            if not isinstance(x[0], Iterable):
                rotation = 0
            elif len(x[0]) == 1:
                rotation = 0
            else:
                rotation = -60

        ax.set_xticks(x_pos)
        ax.set_xticklabels(xticklabels, rotation=rotation)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if norm == 'log':
            norm = mpl.colors.LogNorm()
        c = ax.pcolormesh(X, Y, Z, norm=norm, cmap=self.cmap)

        cb = plt.colorbar(c, ax=ax)
        labelsize = self.custom_rc['axes.labelsize']
        ticksize = self.custom_rc['ytick.labelsize']
        cb.set_label(zlabel, size=0.8 * labelsize)
        cb.ax.tick_params(labelsize=0.8 * ticksize)

        if save_name is None:
            save_name = title
        self.save_fig(fig, save_name)
        # if self.flag_close:
        #     cb.remove()

    def plot_matrix(
        self,
        M: Union[np.ndarray, qp.Qobj],
        M_ideal: Union[np.ndarray, qp.Qobj] = None,
        xlabels=None,
        ylabels=None,
        title=None,
        limits=None,
        phase_limits=None,
        colorbar=True,
        alpha=None,
        fig=None,
        ax=None,
        threshold=None,
        save_name: str = None,
        flag_save=None
    ):
        if isinstance(M, qp.Qobj):
            if M.type in ['ket', 'bra']:
                M = qp.ket2dm(M)
            M = M.full()

        n = np.size(M)
        xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
        xpos = xpos.T.flatten() - 0.5
        ypos = ypos.T.flatten() - 0.5
        zpos = np.zeros(n)
        dx = dy = 0.8 * np.ones(n)
        Mvec = M.flatten()
        dz = abs(Mvec)

        # make small numbers real, to avoid random colors
        idx, = np.where(abs(Mvec) < 0.001)
        Mvec[idx] = abs(Mvec[idx])

        if phase_limits:  # check that limits is a list type
            phase_min = phase_limits[0]
            phase_max = phase_limits[1]
        else:
            phase_min = -np.pi
            phase_max = np.pi

        norm = mpl.colors.Normalize(phase_min, phase_max)
        cmap = qp.complex_phase_cmap()

        colors = cmap(norm(np.angle(Mvec)))
        if threshold is not None:
            colors[:, 3] = 1 * (dz > threshold)

        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig, azim=-35, elev=35)

        alpha = alpha or self.alpha

        ax.bar3d(
            xpos, ypos, zpos, dx, dy, dz,
            color=colors, alpha=alpha
        )

        if M_ideal is not None:
            if isinstance(M_ideal, qp.Qobj):
                if M_ideal.type in ['ket', 'bra']:
                    M_ideal = qp.ket2dm(M_ideal)
                M_ideal = M_ideal.full()

            M_ideal_vec = M_ideal.flatten()
            dz_ideal = abs(M_ideal_vec)

            ax.bar3d(
                xpos, ypos, zpos, dx, dy, dz_ideal,
                color=np.array([[1, 1, 1, 0]] * len(colors)),
                edgecolor='black', linewidth=1.5
            )

        if title:
            ax.set_title(title)

        # x axis
        xtics = -0.5 + np.arange(M.shape[0])
        ax.axes.w_xaxis.set_major_locator(plt.FixedLocator(xtics))
        if xlabels:
            nxlabels = len(xlabels)
            if nxlabels != len(xtics):
                raise ValueError(f"got {nxlabels} xlabels but needed {len(xtics)}")
            ax.set_xticklabels(xlabels)
        ax.tick_params(axis='x', labelsize=12)

        # y axis
        ytics = -0.5 + np.arange(M.shape[1])
        ax.axes.w_yaxis.set_major_locator(plt.FixedLocator(ytics))
        if ylabels:
            nylabels = len(ylabels)
            if nylabels != len(ytics):
                raise ValueError(f"got {nylabels} ylabels but needed {len(ytics)}")
            ax.set_yticklabels(ylabels)
        ax.tick_params(axis='y', labelsize=12)

        # z axis
        if limits and isinstance(limits, list):
            ax.set_zlim3d(limits)
        else:
            ax.set_zlim3d([0, 1])  # use min/max

        # color axis
        if colorbar:
            cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.0)
            cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
            cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cb.set_ticklabels(
                (r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
            cb.set_label('arg')

        if save_name is None:
            save_name = title
        self.save_fig(fig, save_name, flag_save)
        return fig, ax


def normalize_tick_label(arg_name: str, arg, tick_param: dict = None):
    pos_all = np.arange(len(arg))
    num = (
        tick_param.get('num_x', 10) if arg_name == 'x' else tick_param.get('num_y', 10)
    )
    sep = round(len(arg) / num)
    sep = 1 if sep == 0 else sep
    pos = pos_all[::sep]
    # 如果x/y中的元素是数组，则用元组(小括号)的形式展示；如果x/y中的元素是一个数，则按照正常方式展示
    digits = (
        tick_param.get('digits_x', 2)
        if arg_name == 'x'
        else tick_param.get('digits_y', 3)
    )
    ticklabels = [np.around(i, digits).astype(str) for i in arg][::sep]
    if isinstance(ticklabels[0], np.ndarray):
        ticklabels = [', '.join(label) for label in ticklabels]
    return pos, ticklabels


if __name__ == '__main__':
    pass
