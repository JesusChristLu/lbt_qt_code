# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/01
# __author:       HanQing Shi
"""File contains some internal config definitions.
"""
import os

from PySide6.QtCore import Qt

from pyQCat.invoker import DEFAULT_FOLDER
from pyQCat.structures import QDict
from pyQCat.types import LogFormat
from pyQCat.types import StandardContext

GUI_CONFIG = QDict(
    tips=[
        "Clicking the tree libraries to select a experiment to run.",
        "Changed some object parameters? Click the <b>Remake</b> button in the main "
        "toolbar to recreate the polygons.",
        """<b>Log widget:</b> Right click the logger window to be able to change the log level and
        the loggers that are shown/hidden.""",
        """<b>All component widget:</b> Double click a component to zoom into it!""",
    ],
    log_conf=QDict(
        set_levels=[
            QDict(
                name="FLOW",
                no=21,
                color="<cyan><bold>",
                icon="🐍"
            ),
            QDict(
                name="EXP",
                no=22,
                color="<blue>",
                icon="🐍"
            ),
            QDict(
                name="RESULT",
                no=23,
                color="<green>",
                icon="🐍"
            ),
            QDict(
                name="UPDATE",
                no=24,
                color="<yellow><bold>",
                icon="🐍"
            )
        ],
        style="""
        span{font-weight:bold; font-size: 15px}
        .DEBUG {color: #3384d4}
        .WARNING {color: #a68a0d}
        .ERROR {color: #f0524f}
        .INFO {}
        .SUCCESS {font-weight: green;}
        .CRITICAL {font-weight: bold;}
        .TRACE {font-weight: green;}
        .Level_21 {color: #109ea3}
        .Level_22 {color: #317dd4}
        .Level_23 {color: #5a962c}
        .Level_24 {color: #FF00FF}
        .time {color: #a67c1c}
        .sep {color: #f05042}
        .pre {font-family: JetBrains Mono; font-weight: normal;}
        .version{color: #109ea3; font-size: 12px}
        """,
        num_lines=1500,
        level="INFO",
        # todo optimize and unify log format
        format=LogFormat.simple,
        verbose_format=LogFormat.detail,
    ),
    main_window=QDict(
        title="pyQCat Visage - The Quantum Chip Calibration",
        auto_size=False,  # Autosize on creation of window
        style="fusion",
    ),
    graphics_view=QDict(
        theme_classic_dark=QDict(
            color=(25, 35, 45, 255),
            grid_pen_s=(52, 52, 52, 255),
            grid_pen_l=(22, 22, 22, 255),
            grid_size_fine=15,
            grid_size_course=150,
            mouse_wheel_zoom_rate=0.0015,
            qubit_color=(50, 70, 241, 215),
            coupler_color=(230, 244, 255, 255),
            edge_color=(117, 184, 239, 255),
            pen_color=(241, 175, 0, 255),
            env_color=(50, 205, 50, 255),
            physical_color=(183, 55, 70, 255),

            font_color=(255, 255, 255, 255),
            focal_font_color=(255, 20, 147, 255),
            fonts="Lucida Sans Unicode",
            font_size=20,

            port_size=20,
            port_radius=5,
            port_margin=2,
            port_color=(49, 115, 100, 255),  #: default port size.
            port_border_color=(29, 202, 151, 255),  #: default port border color.
            port_active_color=(14, 45, 59, 255),  #: port color when selected.
            port_active_border_color=(107, 166, 193, 255),  #: port border color when selected.
            port_hover_color=(17, 43, 82, 255),  #: port color on mouse over.
            port_hover_border_color=(136, 255, 35, 255),  #: port border color on mouse over.
            port_text_color=(255, 255, 0, 255),

            pipe_color=(0, 128, 255, 255),
            pipe_highlight_color=(255, 102, 0, 255),
            pipe_sub_graph_color=(125, 224, 158, 255),

            node_color=(65, 84, 105, 170),
            node_h_margin=15,
            node_v_margin=10,
            node_running=(0, 191, 255, 200),
            node_success=(0, 255, 1, 200),
            node_failed=(254, 0, 0, 200),
            node_selected=(241, 175, 0, 255),
            thread_color=QDict(
                env_color=(255, 212, 46, 255),
                secure_color = (0, 130, 255, 255),
                alter_color = (21, 242, 108,255),
                use_color = (255, 46, 46, 255),
                t0=QDict(
                    env_color=(82, 172, 255, 150),
                    physical_color=(63, 131, 194, 255),
                ),
                t1=QDict(
                    env_color=(255, 46, 46, 150),
                    physical_color=(237, 43, 47, 255),
                ),
                t2=QDict(
                    env_color=(21, 242, 108, 150),
                    physical_color=(15, 166, 74, 255),
                ),
                t3=QDict(
                    env_color=(255, 212, 46, 150),
                    physical_color=(252, 183, 46, 255),
                ),
                t4=QDict(
                    env_color=(38, 38, 38, 150),
                    physical_color=(62, 0, 0, 255),
                ),
            )
        ),
        theme_classic_light=QDict(
            color=(250, 250, 250, 255),
            grid_pen_s=(229, 229, 229, 255),
            grid_pen_l=(201, 205, 208, 255),
            grid_size_fine=15,
            grid_size_course=150,
            mouse_wheel_zoom_rate=0.0015,

            qubit_color=(50, 70, 241, 215),
            coupler_color=(155, 155, 155, 255),
            edge_color=(117, 184, 239, 170),
            pen_color=(255, 0, 0, 255),
            env_color=(50, 205, 50, 255),
            physical_color=(183, 55, 70, 255),

            font_color=(0, 0, 0, 255),
            focal_font_color=(255, 20, 147, 255),
            fonts="Lucida Sans Unicode",
            font_size=20,

            port_size=20,
            port_radius=5,
            port_margin=2,
            port_color=(49, 115, 100, 255),  #: default port size.
            port_border_color=(29, 202, 151, 255),  #: default port border color.
            port_active_color=(14, 45, 59, 255),  #: port color when selected.
            port_active_border_color=(107, 166, 193, 255),  #: port border color when selected.
            port_hover_color=(17, 43, 82, 255),  #: port color on mouse over.
            port_hover_border_color=(136, 255, 35, 255),  #: port border color on mouse over.
            port_text_color=(255, 255, 0, 255),

            pipe_color=(0, 128, 255, 255),
            pipe_highlight_color=(255, 102, 0, 255),
            pipe_sub_graph_color=(125, 224, 158, 255),

            node_color=(201, 205, 208, 170),
            node_h_margin=15,
            node_v_margin=10,
            node_running=(0, 191, 255, 200),
            node_success=(0, 255, 1, 200),
            node_failed=(254, 0, 0, 200),
            node_selected=(241, 175, 0, 255),
            thread_color=QDict(
                env_color=(255, 212, 46, 255),
                secure_color = (0, 130, 255, 255),
                alter_color = (21, 242, 108,255),
                use_color = (255, 46, 46, 255),
                t0=QDict(
                    env_color=(82, 172, 255, 150),
                    physical_color=(63, 131, 194, 255),
                ),
                t1=QDict(
                    env_color=(255, 46, 46, 150),
                    physical_color=(237, 43, 47, 255),
                ),
                t2=QDict(
                    env_color=(21, 242, 108, 150),
                    physical_color=(15, 166, 74, 255),
                ),
                t3=QDict(
                    env_color=(255, 212, 46, 150),
                    physical_color=(252, 183, 46, 255),
                ),
                t4=QDict(
                    env_color=(38, 38, 38, 150),
                    physical_color=(62, 0, 0, 255),
                ),
            )
        ),
        theme_visage_light=QDict(
            color=(253, 252, 251, 255),
            grid_pen_s=(229, 229, 229, 255),
            grid_pen_l=(201, 205, 208, 255),
            grid_size_fine=15,
            grid_size_course=150,
            mouse_wheel_zoom_rate=0.0015,
            qubit_color=(50, 70, 241, 215),
            coupler_color=(155, 155, 155, 255),
            edge_color=(117, 184, 239, 170),
            pen_color=(255, 0, 0, 255),
            env_color=(50, 205, 50, 255),
            physical_color=(183, 55, 70, 255),

            font_color=(0, 0, 0, 255),
            focal_font_color=(255, 20, 147, 255),
            fonts="Lucida Sans Unicode",
            font_size=20,

            port_size=20,
            port_radius=5,
            port_margin=2,
            port_color=(49, 115, 100, 255),  #: default port size.
            port_border_color=(29, 202, 151, 255),  #: default port border color.
            port_active_color=(14, 45, 59, 255),  #: port color when selected.
            port_active_border_color=(107, 166, 193, 255),  #: port border color when selected.
            port_hover_color=(17, 43, 82, 255),  #: port color on mouse over.
            port_hover_border_color=(136, 255, 35, 255),  #: port border color on mouse over.
            port_text_color=(255, 255, 0, 255),

            pipe_color=(0, 128, 255, 255),
            pipe_highlight_color=(255, 102, 0, 255),
            pipe_sub_graph_color=(125, 224, 158, 255),

            node_color=(203, 203, 203, 255),
            node_h_margin=15,
            node_v_margin=10,
            node_running=(0, 191, 255, 200),
            node_success=(0, 255, 1, 200),
            node_failed=(254, 0, 0, 200),
            node_selected=(241, 175, 0, 255),
            thread_color=QDict(
                env_color=(255, 212, 46, 255),
                secure_color = (0, 130, 255, 255),
                alter_color = (21, 242, 108,255),
                use_color = (255, 46, 46, 255),
                t0=QDict(
                    env_color=(82, 172, 255, 150),
                    physical_color=(63, 131, 194, 255),
                ),
                t1=QDict(
                    env_color=(255, 46, 46, 150),
                    physical_color=(237, 43, 47, 255),
                ),
                t2=QDict(
                    env_color=(21, 242, 108, 150),
                    physical_color=(15, 166, 74, 255),
                ),
                t3=QDict(
                    env_color=(255, 212, 46, 150),
                    physical_color=(252, 183, 46, 255),
                ),
                t4=QDict(
                    env_color=(38, 38, 38, 150),
                    physical_color=(62, 0, 0, 255),
                ),
            )
        ),
        theme_visage_dark=QDict(
            color=(43, 43, 43, 255),
            grid_pen_s=(52, 52, 52, 255),
            grid_pen_l=(22, 22, 22, 255),
            grid_size_fine=15,
            grid_size_course=150,
            mouse_wheel_zoom_rate=0.0015,
            qubit_color=(50, 70, 241, 215),
            coupler_color=(230, 244, 255, 255),
            edge_color=(117, 184, 239, 170),
            pen_color=(241, 175, 0, 255),
            env_color=(50, 205, 50, 255),
            physical_color=(183, 55, 70, 255),

            font_color=(255, 255, 255, 255),
            focal_font_color=(255, 20, 147, 255),
            fonts="Lucida Sans Unicode",
            font_size=20,

            port_size=20,
            port_radius=5,
            port_margin=2,
            port_color=(49, 115, 100, 255),  #: default port size.
            port_border_color=(29, 202, 151, 255),  #: default port border color.
            port_active_color=(14, 45, 59, 255),  #: port color when selected.
            port_active_border_color=(107, 166, 193, 255),  #: port border color when selected.
            port_hover_color=(17, 43, 82, 255),  #: port color on mouse over.
            port_hover_border_color=(136, 255, 35, 255),  #: port border color on mouse over.
            port_text_color=(255, 255, 0, 255),

            pipe_color=(0, 128, 255, 255),
            pipe_highlight_color=(255, 102, 0, 255),
            pipe_sub_graph_color=(125, 224, 158, 255),

            node_color=(60, 63, 65, 255),
            node_h_margin=15,
            node_v_margin=10,
            node_running=(0, 191, 255, 200),
            node_success=(0, 255, 1, 200),
            node_failed=(254, 0, 0, 200),
            node_selected=(241, 175, 0, 255),
            thread_color=QDict(
                env_color=(255, 212, 46, 255),
                secure_color = (0, 130, 255, 255),
                alter_color = (21, 242, 108,255),
                use_color = (255, 46, 46, 255),
                t0=QDict(
                    env_color=(82, 172, 255, 150),
                    physical_color=(63, 131, 194, 255),
                ),
                t1=QDict(
                    env_color=(255, 46, 46, 150),
                    physical_color=(237, 43, 47, 255),
                ),
                t2=QDict(
                    env_color=(21, 242, 108, 150),
                    physical_color=(15, 166, 74, 255),
                ),
                t3=QDict(
                    env_color=(255, 212, 46, 150),
                    physical_color=(252, 183, 46, 255),
                ),
                t4=QDict(
                    env_color=(38, 38, 38, 150),
                    physical_color=(62, 0, 0, 255),
                ),
            )
        ),
    ),
    dynamic=QDict(
        y0_color=Qt.red,
        y1_color=Qt.yellow,
        theme=1,
    ),
    chart_style=QDict(
        cs0=("chart_light_theme", "actionLight"),
        cs1=("chart_bc_theme", "actionBlue_Cerulean"),
        cs2=("chart_dark_theme", "actionDark"),
        cs3=("chart_bs_theme", "actionBrown_Sand"),
        cs4=("chart_bn_theme", "actionBlue_NVS"),
        cd5=("chart_hc_theme", "actionHigh_Contrast"),
        cd6=("chart_bi_theme", "actionBlue_Icy"),
        cs7=("chart_qt_theme", "actionQt"),
    ),
    report_theme_map=QDict(
        default='light',
        darkstyle='dark',
        visage_dark='dark',
        visage_light='light'
    ),
    component_icon=QDict(
        qubit=u":/qubit.png",
        coupler=u":/coupler.png",
        json=u":/cpu.png",
        dat=u":/compensate.png",
        bin=u":/dcm.png",
        qubit_pair=u":/pair.png"
    ),
    component_sup_delete=[
        "union_rd.json",
        "xy_crosstalk.json",
        "hardware_offset.json"
    ],
    cache_file_name=QDict(
        config='config.conf',
        std_context='std_context.json'
    ),
    cache_user_bin=os.path.join(DEFAULT_FOLDER, 'cache_user.bin'),
    cache_user_path=os.path.join(DEFAULT_FOLDER, '.user'),
    file_icon=QDict(
        folder=u":/folder.png",
        dat=u":/dat.png",
        png=u":/png.png",
        txt=u":/txt.png",
        log=u':/log.png',
        json=u':/json.png'
    ),
    multi_box_row_height=25,
    document_schedule=QDict(
        pen_width=1.5,
        x_label="time(ns)",
        y_label="Amp(v)",
        x_tick_count=6,
        y_tick_count=5,
        color_lines=[
            Qt.GlobalColor.darkGreen,
            Qt.GlobalColor.darkRed,
            Qt.GlobalColor.darkYellow,
            Qt.GlobalColor.darkCyan,
            Qt.GlobalColor.darkBlue
        ]
    ),
    std_context=QDict(
        qubit_calibration=QDict(
            default="",
            f01="01",
            f02="02",
            f012="012"
        ),
        coupler_probe_calibration=QDict(
            default="",
            f01="01",
            f02="02",
            f012="012"
        ),
        union_read_measure=QDict(
            default="",
            f01="01",
            f02="02",
            f012="012"
        ),
        coupler_calibration=QDict(
            Coupler="Coupler",
            probeQ="probeQ",
            driveQ="driveQ",
            QH="QH",
            QL="QL",
        ),
        cz_gate_calibration=QDict(
            ql_01="ql-01",
            ql_012="ql-012",
            qh_01="qh-01",
            qh_012="qh-012",
            union_01_01="union-01-01",
            union_012_01="union-012-01",
            union_012_012="union-012-012"
        )
    ),
    context_map=QDict(
        ReadoutSampleDelayCalibrate=[StandardContext.QC.value],
        AmpOptimize=[StandardContext.QC.value, StandardContext.CPC.value],
        CouplerAmpOptimize=[StandardContext.CC.value],
        APE=[StandardContext.QC.value, StandardContext.CPC.value],
        CouplerAPE=[StandardContext.CC.value],
        CavityFreqSpectrum=[StandardContext.QC.value],
        CZAssist=[StandardContext.CGC.value],
        DistortionT1=[StandardContext.QC.value],
        CouplerDistortionT1=[StandardContext.CC.value],
        QubitSpectrum=[StandardContext.QC.value, StandardContext.CPC.value],
        CouplerSpectrum=[StandardContext.CC.value],
        RabiScanAmp=[StandardContext.QC.value, StandardContext.CPC.value],
        RabiScanWidth=[StandardContext.QC.value, StandardContext.CPC.value],
        CouplerRabiScanAmp=[StandardContext.CC.value],
        CouplerRabiScanWidth=[StandardContext.CC.value],
        Ramsey=[StandardContext.QC.value, StandardContext.CPC.value],
        RamseyCrosstalk=[StandardContext.CM.value],
        CouplerRamsey=[StandardContext.CC.value],
        CouplerRamseyCrosstalk=[StandardContext.CM.value],
        RBSingle=[StandardContext.QC.value],
        JointRBSingle=[StandardContext.CGC.value],
        RBMultiple=[StandardContext.CGC.value],
        SingleShot=[StandardContext.QC.value, StandardContext.CPC.value],
        SwapOnce=[StandardContext.CGC.value],
        T1=[StandardContext.QC.value],
        CouplerT1=[StandardContext.CC.value],
        XYZTiming=[StandardContext.QC.value],
        CouplerXYZTiming=[StandardContext.CC.value],
        CouplerXYZTimingByZZShift=[StandardContext.CC.value],
        UnionReadout=[StandardContext.URM.value],
        UnionReadoutAlter=[StandardContext.URM.value],
        SingleShotF012=[StandardContext.QC.value],
        SingleShotF02=[StandardContext.QC.value],
        CavitySpectrumF12=[StandardContext.QC.value],
        RabiScanWidthF12=[StandardContext.QC.value],
        RabiScanAmpF12=[StandardContext.QC.value],
        RamseyF12=[StandardContext.QC.value],
        QubitSpectrumF12=[StandardContext.QC.value],
        ACCvaluerosstalkOnce=[StandardContext.CM.value],
        LeakageOnce=[StandardContext.CGC.value],
        StateTomography=[StandardContext.QC.value, StandardContext.CGC.value],
        Distortion_RB=[StandardContext.QC.value],
        RamseyZZ=[StandardContext.CGC.value],
        SpinEchoZZ=[StandardContext.CGC.value],
        SpinEcho=[StandardContext.QC.value],
        ZZTimingOnce=[StandardContext.CGC.value],
        ZZTimingComposite=[StandardContext.CGC.value],
        CPhaseTMSE=[StandardContext.CGC.value],
        SQPhaseTMSE=[StandardContext.CGC.value],
        ConditionalPhaseTMSE=[StandardContext.CGC.value],
        ConditionalPhaseTMSENGate=[StandardContext.CGC.value],
        XEBSingle=[StandardContext.QC.value],
        XEBMultiple=[StandardContext.CGC.value],
        ACCrosstalk=[StandardContext.CM.value],
        ACSpectrum=[StandardContext.QC.value],
        APEComposite=[StandardContext.QC.value, StandardContext.CPC.value],
        CouplerAPEComposite=[StandardContext.CC.value],
        CouplerDetuneCalibration=[StandardContext.CC.value],
        ConditionalPhaseFixed=[StandardContext.CGC.value],
        ConditionalPhaseAdjust=[StandardContext.CGC.value],
        ConditionalPhaseAdjustNGate=[StandardContext.CGC.value],
        DCCrosstalk=[StandardContext.CM.value],
        DCSpectrumSpec=[StandardContext.QC.value],
        DistortionT1Composite=[StandardContext.QC.value],
        QubitSpectrumComposite=[StandardContext.QC.value],
        ReadoutFreqCalibrate=[StandardContext.QC.value],
        ReadoutFreqSSCalibrate=[StandardContext.QC.value],
        ReadoutPowerCalibrate=[StandardContext.QC.value],
        SampleWidthOptimize=[StandardContext.QC.value],
        SingleQubitPhase=[StandardContext.CGC.value],
        Swap=[StandardContext.CGC.value],
        T1Spectrum=[StandardContext.QC.value],
        T2Spectrum=[StandardContext.QC.value],
        CouplerT1Spectrum=[StandardContext.CC.value],
        QubitFreqCalibration=[StandardContext.QC.value, StandardContext.CPC.value],
        CouplerFreqCalibration=[StandardContext.CC.value],
        DetuneCalibration=[StandardContext.QC.value, StandardContext.CPC.value],
        ProcessTomography=[StandardContext.QC.value, StandardContext.CGC.value],
        XpiDetection=[StandardContext.QC.value, StandardContext.CPC.value],
        CavityTunable=[StandardContext.QC.value, StandardContext.CC.value],
        F12Calibration=[StandardContext.QC.value],
        ReadoutPowerF02Calibrate=[StandardContext.QC.value],
        CavityShiftF012=[StandardContext.QC.value],
        ACCrosstalkFixF=[StandardContext.CM.value],
        CouplerACSpectrum=[StandardContext.CC.value],
        LeakageAmp=[StandardContext.CGC.value],
        LeakageNum=[StandardContext.CGC.value],
        T2Ramsey=[StandardContext.QC.value],
        CouplerT2Ramsey=[StandardContext.CC.value],
        CouplerTunableByQS=[StandardContext.CC.value],
        QubitSpectrumZAmp=[StandardContext.QC.value],
        CouplerSpectrumZAmp=[StandardContext.CC.value],
        CavityPowerScan=[StandardContext.QC.value],
        DistortionPolesOpt=[StandardContext.QC.value],
        RBInterleavedMultiple=[StandardContext.CGC.value],
        ZZShiftRamsey=[StandardContext.CGC.value],
        ZZShiftSpinEcho=[StandardContext.CGC.value],
        StabilityT1=[StandardContext.QC.value],
        StabilityT2Ramsey=[StandardContext.QC.value],
        StabilitySingleShot=[StandardContext.QC.value],
        SweepDetuneRabiWidth=[StandardContext.QC.value],
        CouplerDistortionZZComposite=[StandardContext.CC.value],
        CouplerDistortionT1Composite=[StandardContext.CC.value],
        OptimizeFIR=[StandardContext.QC.value],
        CouplerOptimizeFIR=[StandardContext.CC.value],
        CouplerZZOptimizeFIR=[StandardContext.CC.value],
        NMRBMultiple=[StandardContext.CGC.value],
        NMRBSingle=[StandardContext.QC.value],
        NMSingleShot=[StandardContext.QC.value],
        NMXEBSingle=[StandardContext.QC.value],
        NMXEBMultiple=[StandardContext.CGC.value],
        AmpComposite=[StandardContext.QC.value],
        CouplerAmpComposite=[StandardContext.CC.value],
        VoltageDriftOneStepCalibration=[StandardContext.QC.value],
        VoltageDriftGradientCalibration=[StandardContext.QC.value],
        SweetPointCalibration=[StandardContext.QC.value],
        SwapSweetPointCalibration=[StandardContext.CGC.value],
        ZZShiftSweetPointCalibration=[StandardContext.CC.value],
        FindBusCavityFreq=[StandardContext.QC.value, StandardContext.URM.value],
        CavityFluxScan=[StandardContext.QC.value],
        RoomTempDistortion=[StandardContext.QC.value],
        QCShift=[StandardContext.CM.value],
        QCT1=[StandardContext.CPC.value],
        QCT1Spectrum=[StandardContext.CPC.value],
        CouplerRabiScanWidthDetune=[StandardContext.CC.value],
        CouplerSweepDetuneRabiWidth=[StandardContext.CC.value],
        PurityRBSingle=[StandardContext.QC.value],
        PurityRBMultiple=[StandardContext.CGC.value],
        PurityRBInterleavedSingle=[StandardContext.QC.value],
        PurityRBInterleavedMultiple=[StandardContext.CGC.value],
        XYCrossRabiWidth=[StandardContext.URM.value, StandardContext.CM.value],
        XYCrossPlusRabiWidth=[StandardContext.URM.value, StandardContext.CM.value],
    ),
    help_url=QDict(
        community="https://document.qpanda.cn/space/9030MdOBwNfe5oqw",
        user_guide="https://document.qpanda.cn/docs/913JVW2NKdsD8B3E",
        exp_indexes="https://document.qpanda.cn/docs/erAdP6KB7VfZd5AG",
        exp_lib=QDict(
            RBSingle="https://document.qpanda.cn/docs/8Nk6MwZEddTRJGqL",
            RBMultiple="https://document.qpanda.cn/docs/8Nk6MwZEddTRJGqL",
            JointRBSingle="https://document.qpanda.cn/docs/8Nk6MwZEddTRJGqL",
            XEBSingle="https://document.qpanda.cn/docs/B1Aw16x4QZsb2oqm",
            XEBMultiple="https://document.qpanda.cn/docs/B1Aw16x4QZsb2oqm",
            ZZTiming="https://document.qpanda.cn/docs/pmkxQYxrpwizDaAN",
            SweepDetuneRabiWidth="https://document.qpanda.cn/docs/9030Mderv8uZ6lqw"
        ),
        question_pool="https://document.qpanda.cn/folder/913JVW9mNPHvY3E6"
    ),
    cz_pulse_types=[
        "FlatTopGaussian",
        "Constant",
        "GaussianSquare",
        "FlatTop",
        "FlatTopGaussianDetune",
        "Slepian",
    ],
)


class UserForbidden:
    change_chip_params = ["m_lo", "xy_lo", "bus", "z_dc_channel", "z_flux_channel", "xy_channel", "readout_channel"]
