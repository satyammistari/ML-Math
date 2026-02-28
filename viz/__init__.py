"""viz package — all three visualization layers for mlmath."""
from viz.ascii_plots import (scatter, line_plot, multi_line, comp_graph,
                              heatmap, gridworld, neural_net_diagram)
from viz.terminal_plots import (loss_curve, distribution_plot, multi_distribution,
                                 scatter_plot, multi_loss, bar, histogram,
                                 roc_curve_plot, eigenvalue_spectrum, scree_plot,
                                 activation_plot, lr_schedule_plot, mcmc_trace,
                                 convergence_plot)
from viz.matplotlib_plots import (heatmap as mpl_heatmap,
                                   scatter_classes, decision_boundary,
                                   loss_surface_3d, contour_gradient,
                                   bias_variance_curves, roc_curve_mpl,
                                   confusion_matrix_plot, calibration_plot,
                                   gmm_ellipses, positional_encoding_heatmap,
                                   attention_heatmap, radar_chart,
                                   pca_scatter, gp_posterior,
                                   learning_curves, poly_fit_demo)
