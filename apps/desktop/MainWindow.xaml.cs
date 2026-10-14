using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.Win32;

namespace SwiftMev;

public partial class MainWindow : Window
{
    // Estado do Projeto e Imagens
    private string? _loadedImagePath;
    private string? _loadedMaskAutoPath;
    private BitmapSource? _sourceBitmap;
    private WriteableBitmap? _editableMaskBitmap;
    private byte[]? _originalMaskBytes;
    private int _imgWidth;
    private int _imgHeight;
    private double _umPerPixel = 0.0;
    private bool _isCalibrated = false;
    private readonly System.Collections.Generic.Stack<byte[]> _undoStack = new();
    private JsonElement _lastMetricsJson;

    // Estado do Pan e Zoom
    private bool _isPanning;
    private Point _panStartMouse;
    private Point _panStartTranslate;

    // Estado da Ferramenta de Régua
    private bool _isRulerActive;
    private Point? _rulerStartPoint;

    // Estado de Edição Manual
    private bool _isDrawing;
    private int _brushRadius = 10;

    [System.Runtime.InteropServices.DllImport("dwmapi.dll", PreserveSig = true)]
    private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int attrValue, int attrSize);
    private const int DWMWA_USE_IMMERSIVE_DARK_MODE = 20;

    public MainWindow()
    {
        InitializeComponent();
        Loaded += MainWindow_Loaded;
        SourceInitialized += MainWindow_SourceInitialized;
        UpdateCalibrationDisplay();
    }

    private bool _isDarkMode = false;

    private void MainWindow_SourceInitialized(object? sender, EventArgs e)
    {
        SetDwmTheme(_isDarkMode);
    }

    private void ThemeToggle_Click(object sender, RoutedEventArgs e)
    {
        SetTheme(!_isDarkMode);
    }

    private void SetTheme(bool dark)
    {
        _isDarkMode = dark;
        if (dark)
        {
            Resources["WindowBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0B0F17"));
            Resources["TopBarBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0F172A"));
            Resources["BorderSubtleBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1E293B"));
            Resources["CardBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0F172A"));
            Resources["SubCardBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1E293B"));
            Resources["TextPrimaryBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F1F5F9"));
            Resources["TextSecondaryBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#94A3B8"));
            Resources["BrandTitleBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#38BDF8"));
            Resources["CanvasBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#030712"));
            Resources["ViewerToolbarBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0F172A"));
            Resources["DropZoneBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0A101D"));
            Resources["TabInactiveBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1E293B"));
            Resources["TabActiveBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0F172A"));
            Resources["InputBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0B0F17"));
            Resources["InputFgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F8FAFC"));
            Resources["BtnSecondaryBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1E293B"));
            Resources["BtnSecondaryFgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E2E8F0"));
            Resources["BtnSecondaryBorderBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#334155"));
            Resources["AuditBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1E293B"));
            Resources["AuditFgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#94A3B8"));
            Resources["ScaleBarLineBrush"] = new SolidColorBrush(Colors.White);
            Resources["ScaleBarTextBrush"] = new SolidColorBrush(Colors.White);
            Resources["ScaleBarOverlayBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#99000000"));

            if (BtnThemeToggle != null) BtnThemeToggle.Content = "☀️ Modo Claro";
            SetDwmTheme(true);
        }
        else
        {
            Resources["WindowBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F4F7FB"));
            Resources["TopBarBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FFFFFF"));
            Resources["BorderSubtleBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#D9E2EC"));
            Resources["CardBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FFFFFF"));
            Resources["SubCardBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F8FAFC"));
            Resources["TextPrimaryBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1F2937"));
            Resources["TextSecondaryBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#5B6B7A"));
            Resources["BrandTitleBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1565C0"));
            Resources["CanvasBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#EAEFF5"));
            Resources["ViewerToolbarBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F8FAFC"));
            Resources["DropZoneBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FFFFFF"));
            Resources["TabInactiveBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E2E8F0"));
            Resources["TabActiveBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FFFFFF"));
            Resources["InputBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FFFFFF"));
            Resources["InputFgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1F2937"));
            Resources["BtnSecondaryBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FFFFFF"));
            Resources["BtnSecondaryFgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1F2937"));
            Resources["BtnSecondaryBorderBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#D9E2EC"));
            Resources["AuditBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F8FAFC"));
            Resources["AuditFgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#334155"));
            Resources["ScaleBarLineBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0F172A"));
            Resources["ScaleBarTextBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#0F172A"));
            Resources["ScaleBarOverlayBgBrush"] = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#CCFFFFFF"));

            if (BtnThemeToggle != null) BtnThemeToggle.Content = "🌙 Modo Escuro";
            SetDwmTheme(false);
        }
        UpdateCalibrationDisplay();
    }

    private void SetDwmTheme(bool dark)
    {
        try
        {
            var handle = new System.Windows.Interop.WindowInteropHelper(this).Handle;
            if (handle != IntPtr.Zero)
            {
                int val = dark ? 1 : 0;
                DwmSetWindowAttribute(handle, DWMWA_USE_IMMERSIVE_DARK_MODE, ref val, sizeof(int));
                DwmSetWindowAttribute(handle, 19, ref val, sizeof(int));
            }
        }
        catch { }
    }

    private void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        var workArea = SystemParameters.WorkArea;
        // Assegura que a janela cabe com folga dentro da área útil da tela (ex: monitores 1366x768)
        if (Width > workArea.Width - 30) Width = Math.Max(800, workArea.Width - 30);
        if (Height > workArea.Height - 30) Height = Math.Max(480, workArea.Height - 30);

        Left = Math.Max(workArea.Left + 10, workArea.Left + (workArea.Width - Width) / 2);
        Top = Math.Max(workArea.Top + 10, workArea.Top + (workArea.Height - Height) / 2);
    }

    // ─────────────────────────────────────────────────────────────
    // CONTROLE DA JANELA (ARRASTAR & BOTÕES DE FECHAR/MAXIMIZAR)
    // ─────────────────────────────────────────────────────────────

    private void TopBar_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (e.ClickCount == 2)
        {
            WindowState = WindowState == WindowState.Maximized ? WindowState.Normal : WindowState.Maximized;
        }
        else if (e.LeftButton == MouseButtonState.Pressed)
        {
            if (WindowState == WindowState.Maximized)
            {
                var point = PointToScreen(e.GetPosition(this));
                WindowState = WindowState.Normal;
                Left = Math.Max(0, point.X - (Width / 2));
                Top = Math.Max(0, point.Y - 20);
            }
            try
            {
                DragMove();
            }
            catch
            {
                // Proteção contra liberação instantânea de clique
            }
        }
    }

    private void Window_DragOver(object sender, DragEventArgs e)
    {
        if (e.Data.GetDataPresent(DataFormats.FileDrop))
            e.Effects = DragDropEffects.Copy;
        else
            e.Effects = DragDropEffects.None;
        e.Handled = true;
    }

    private void Window_Drop(object sender, DragEventArgs e)
    {
        if (e.Data.GetDataPresent(DataFormats.FileDrop))
        {
            var files = (string[])e.Data.GetData(DataFormats.FileDrop);
            if (files != null && files.Length > 0)
            {
                var file = files[0];
                var ext = System.IO.Path.GetExtension(file).ToLowerInvariant();
                if (ext is ".png" or ".jpg" or ".jpeg" or ".tif" or ".tiff" or ".bmp")
                {
                    LoadImageFile(file);
                }
                else
                {
                    ShowWarning($"Formato não suportado: {ext}. Formatos aceitos: PNG, JPG, TIFF, BMP.");
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 1. CARREGAMENTO DE IMAGEM & INICIALIZAÇÃO
    // ─────────────────────────────────────────────────────────────

    private void ChooseImage_Click(object sender, RoutedEventArgs e)
    {
        var picker = new OpenFileDialog
        {
            Title = "Selecionar Imagem Científica (MEV / Microscopia)",
            Filter = "Imagens Científicas|*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp|Todos os Arquivos|*.*"
        };
        if (picker.ShowDialog() == true)
        {
            LoadImageFile(picker.FileName);
        }
    }

    private void LoadImageFile(string filePath)
    {
        try
        {
            _loadedImagePath = filePath;
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.UriSource = new Uri(filePath, UriKind.Absolute);
            bitmap.EndInit();
            bitmap.Freeze();

            _imgWidth = bitmap.PixelWidth;
            _imgHeight = bitmap.PixelHeight;

            // Se o arquivo contiver metadados de DPI diferentes de 96 (comum em TIFFs científicos de 200/300 DPI),
            // normalizamos a representação de tela para 96 DPI exatos, garantindo perfeito alinhamento pixel a pixel
            // entre a imagem original, a máscara WriteableBitmap e a camada de contornos vetoriais.
            if (bitmap.DpiX != 96.0 || bitmap.DpiY != 96.0)
            {
                int stride = (_imgWidth * bitmap.Format.BitsPerPixel + 7) / 8;
                byte[] pixelBuffer = new byte[stride * _imgHeight];
                bitmap.CopyPixels(pixelBuffer, stride, 0);
                _sourceBitmap = BitmapSource.Create(_imgWidth, _imgHeight, 96, 96, bitmap.Format, bitmap.Palette, pixelBuffer, stride);
                _sourceBitmap.Freeze();
            }
            else
            {
                _sourceBitmap = bitmap;
            }

            ImgSource.Source = _sourceBitmap;
            ImgSource.Width = _imgWidth;
            ImgSource.Height = _imgHeight;

            ImageContainer.Width = _imgWidth;
            ImageContainer.Height = _imgHeight;

            // Inicializar camada de máscara vazia (32bpp PBRA)
            _editableMaskBitmap = new WriteableBitmap(_imgWidth, _imgHeight, 96, 96, PixelFormats.Bgra32, null);
            ImgMaskLayer.Source = _editableMaskBitmap;
            ImgMaskLayer.Width = _imgWidth;
            ImgMaskLayer.Height = _imgHeight;

            ImgOverlay.Source = null;
            ImgOverlay.Width = _imgWidth;
            ImgOverlay.Height = _imgHeight;

            TxtEmptyPlaceholder.Visibility = Visibility.Collapsed;
            if (DropZoneBorder != null) DropZoneBorder.Visibility = Visibility.Collapsed;
            WarningBox.Visibility = Visibility.Collapsed;

            ZoomFit();
            UpdateScaleBar();

            StatusText.Text = $"Imagem carregada: {System.IO.Path.GetFileName(filePath)} ({_imgWidth}x{_imgHeight} px)";
            LogAudit($"Imagem importada com sucesso: {filePath}\nDimensões: {_imgWidth} x {_imgHeight} pixels.");
        }
        catch (Exception ex)
        {
            ShowWarning($"Falha ao carregar arquivo de imagem: {ex.Message}");
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 2. ZOOM, PAN E NAVEGAÇÃO MICROSCÓPICA
    // ─────────────────────────────────────────────────────────────

    private void Viewport_MouseWheel(object sender, MouseWheelEventArgs e)
    {
        if (_sourceBitmap == null) return;

        double zoomFactor = e.Delta > 0 ? 1.2 : (1.0 / 1.2);
        ApplyZoom(zoomFactor, e.GetPosition(ViewportCanvas));
        e.Handled = true;
    }

    private void ApplyZoom(double factor, Point center)
    {
        double currentScale = ViewportScale.ScaleX;
        double newScale = Math.Clamp(currentScale * factor, 0.05, 40.0);

        // Zoom centrado na posição do cursor
        double mouseX = center.X;
        double mouseY = center.Y;

        ViewportTranslate.X = mouseX - (mouseX - ViewportTranslate.X) * (newScale / currentScale);
        ViewportTranslate.Y = mouseY - (mouseY - ViewportTranslate.Y) * (newScale / currentScale);

        ViewportScale.ScaleX = newScale;
        ViewportScale.ScaleY = newScale;

        TxtZoomLevel.Text = $"{newScale * 100:F0}%";
        UpdateScaleBar();
    }

    private void ZoomIn_Click(object sender, RoutedEventArgs e)
    {
        if (_sourceBitmap == null) return;
        ApplyZoom(1.25, new Point(ViewportCanvas.ActualWidth / 2, ViewportCanvas.ActualHeight / 2));
    }

    private void ZoomOut_Click(object sender, RoutedEventArgs e)
    {
        if (_sourceBitmap == null) return;
        ApplyZoom(1.0 / 1.25, new Point(ViewportCanvas.ActualWidth / 2, ViewportCanvas.ActualHeight / 2));
    }

    private void ZoomFit_Click(object sender, RoutedEventArgs e) => ZoomFit();

    private void ZoomReset_Click(object sender, RoutedEventArgs e)
    {
        ViewportScale.ScaleX = 1.0;
        ViewportScale.ScaleY = 1.0;
        ViewportTranslate.X = 0;
        ViewportTranslate.Y = 0;
        TxtZoomLevel.Text = "100%";
        UpdateScaleBar();
    }

    private void ZoomFit()
    {
        if (_sourceBitmap == null || ViewportCanvas.ActualWidth <= 0 || ViewportCanvas.ActualHeight <= 0) return;

        double scaleX = (ViewportCanvas.ActualWidth - 40) / _imgWidth;
        double scaleY = (ViewportCanvas.ActualHeight - 40) / _imgHeight;
        double scale = Math.Min(scaleX, scaleY);
        scale = Math.Clamp(scale, 0.05, 5.0);

        ViewportScale.ScaleX = scale;
        ViewportScale.ScaleY = scale;
        ViewportTranslate.X = (ViewportCanvas.ActualWidth - _imgWidth * scale) / 2.0;
        ViewportTranslate.Y = (ViewportCanvas.ActualHeight - _imgHeight * scale) / 2.0;

        TxtZoomLevel.Text = $"{scale * 100:F0}%";
        UpdateScaleBar();
    }

    private void Viewport_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
    {
        _isPanning = true;
        _panStartMouse = e.GetPosition(ViewportCanvas);
        _panStartTranslate = new Point(ViewportTranslate.X, ViewportTranslate.Y);
        ViewportCanvas.CaptureMouse();
        ViewportCanvas.Cursor = Cursors.SizeAll;
    }

    private void Viewport_MouseRightButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (_isPanning)
        {
            _isPanning = false;
            ViewportCanvas.ReleaseMouseCapture();
            UpdateCursor();
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 3. FERRAMENTA DE CALIBRAÇÃO POR BARRA DE ESCALA (RÉGUA)
    // ─────────────────────────────────────────────────────────────

    private void RulerCalibrate_Click(object sender, RoutedEventArgs e)
    {
        if (_sourceBitmap == null)
        {
            ShowWarning("Abra uma imagem antes de iniciar a calibração por régua.");
            return;
        }

        _isRulerActive = true;
        _rulerStartPoint = null;
        RulerLine.Visibility = Visibility.Collapsed;
        RulerStatus.Visibility = Visibility.Visible;
        RulerStatus.Text = "Clique no início da barra de escala...";
        ViewportCanvas.Cursor = Cursors.Cross;
    }

    private void FinishRulerCalibration(Point p1, Point p2)
    {
        double dx = p2.X - p1.X;
        double dy = p2.Y - p1.Y;
        double lengthPx = Math.Sqrt(dx * dx + dy * dy);

        if (lengthPx < 3.0)
        {
            ShowWarning("Segmento muito curto para calibração confiável. Meça a extensão completa da barra.");
            CancelRuler();
            return;
        }

        // Diálogo simples para entrada de microns
        var inputPrompt = Microsoft.VisualBasic.Interaction.InputBox(
            $"A régua mediu {lengthPx:F1} pixels.\n\nInforme o valor real correspondente em micrômetros (μm):",
            "Calibração por Barra de Escala",
            "10.0"
        );

        if (double.TryParse(inputPrompt.Replace(',', '.'), NumberStyles.Float, CultureInfo.InvariantCulture, out var lengthUm) && lengthUm > 0)
        {
            _umPerPixel = lengthUm / lengthPx;
            _isCalibrated = true;
            TxtCalibration.Text = _umPerPixel.ToString("0.######", CultureInfo.InvariantCulture);
            UpdateCalibrationDisplay();
            LogAudit($"Calibração por régua aplicada: {lengthPx:F1} px = {lengthUm:F2} μm => Escala: {_umPerPixel:F6} μm/px.");
            StatusText.Text = $"Calibração definida: {_umPerPixel:F4} μm/px";
            UpdateScaleBar();
        }
        else
        {
            ShowWarning("Entrada de calibração cancelada ou inválida.");
        }

        CancelRuler();
    }

    private void CancelRuler()
    {
        _isRulerActive = false;
        _rulerStartPoint = null;
        RulerLine.Visibility = Visibility.Collapsed;
        RulerStatus.Visibility = Visibility.Collapsed;
        UpdateCursor();
    }

    private void Calibration_TextChanged(object sender, TextChangedEventArgs e)
    {
        if (!IsLoaded || TxtCalibration == null) return;
        if (double.TryParse(TxtCalibration.Text.Trim().Replace(',', '.'), NumberStyles.Float, CultureInfo.InvariantCulture, out var calib) && calib > 0)
        {
            _umPerPixel = calib;
            _isCalibrated = true;
            UpdateCalibrationDisplay();
            UpdateScaleBar();
        }
        else
        {
            _isCalibrated = false;
            UpdateCalibrationDisplay();
        }
    }

    private void ConfirmCalibration_Click(object sender, RoutedEventArgs e)
    {
        var text = TxtCalibration.Text.Trim().Replace(',', '.');
        if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var cal) && cal > 0)
        {
            _umPerPixel = cal;
            _isCalibrated = true;
            UpdateCalibrationDisplay();
            UpdateScaleBar();
            WarningBox.Visibility = Visibility.Collapsed;
            LogAudit($"Calibração espacial confirmada manualmente: {_umPerPixel:F6} μm/pixel.");
            StatusText.Text = $"Calibração definida: {_umPerPixel:F4} μm/px";
        }
        else
        {
            ShowWarning("Insira um valor numérico válido maior que zero (ex: 0.5 ou 1.25) para confirmar a escala.");
        }
    }

    private void GoToCalibration_Click(object sender, RoutedEventArgs e)
    {
        if (TabCalibration != null) TabCalibration.IsSelected = true;
    }

    private void UpdateCalibrationDisplay()
    {
        if (BadgeCalibStatusAnalyse != null)
        {
            if (_isCalibrated && _umPerPixel > 0)
            {
                BadgeCalibStatusAnalyse.Text = $"✓ Calibrado: {_umPerPixel:F4} μm/px";
                BadgeCalibStatusAnalyse.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#047857"));
            }
            else
            {
                BadgeCalibStatusAnalyse.Text = "⚠️ Calibração Pendente";
                BadgeCalibStatusAnalyse.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#B45309"));
            }
        }
        if (BadgeCalibStatusTab != null)
        {
            if (_isCalibrated && _umPerPixel > 0)
            {
                BadgeCalibStatusTab.Text = $"✓ Escala Confirmada: {_umPerPixel:F4} μm/px";
                BadgeCalibStatusTab.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#047857"));
            }
            else
            {
                BadgeCalibStatusTab.Text = "⚠️ Escala Não Definida";
                BadgeCalibStatusTab.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E5484D"));
            }
        }
    }

    private void UpdateScaleBar()
    {
        if (!IsLoaded || ScaleBarOverlay == null || ScaleBarLine == null || ScaleBarLabel == null || ViewportScale == null) return;
        if (_sourceBitmap == null || _umPerPixel <= 0 || ChkShowScaleBar?.IsChecked != true)
        {
            ScaleBarOverlay.Visibility = Visibility.Collapsed;
            return;
        }

        // Calcular tamanho elegante para a barra (entre 60 e 180 pixels na tela)
        double currentZoom = ViewportScale.ScaleX;
        double pxTarget = 100.0;
        double umTarget = (pxTarget / currentZoom) * _umPerPixel;

        // Arredondar para número redondo em micrometros (ex: 1, 2, 5, 10, 20, 50, 100, 200...)
        double roundUm = RoundToNiceScale(umTarget);
        double actualPx = (roundUm / _umPerPixel) * currentZoom;

        ScaleBarLine.Width = actualPx;
        ScaleBarLabel.Text = $"{roundUm} μm";
        ScaleBarOverlay.Visibility = Visibility.Visible;
    }

    private static double RoundToNiceScale(double value)
    {
        if (value <= 0) return 1.0;
        double exponent = Math.Floor(Math.Log10(value));
        double fraction = value / Math.Pow(10, exponent);
        double niceFraction = fraction switch
        {
            < 1.5 => 1.0,
            < 3.0 => 2.0,
            < 7.0 => 5.0,
            _ => 10.0
        };
        return niceFraction * Math.Pow(10, exponent);
    }

    // ─────────────────────────────────────────────────────────────
    // 4. EDIÇÃO MANUAL DE MÁSCARAS (PINCEL & BORRACHA)
    // ─────────────────────────────────────────────────────────────

    private void Tool_Checked(object sender, RoutedEventArgs e)
    {
        if (IsLoaded) UpdateCursor();
    }

    private void SliderBrushSize_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        _brushRadius = (int)e.NewValue;
        if (LblBrushSize != null) LblBrushSize.Text = $"{_brushRadius} px";
    }

    private void UpdateCursor()
    {
        if (_isRulerActive) ViewportCanvas.Cursor = Cursors.Cross;
        else if (ToolBrush.IsChecked == true) ViewportCanvas.Cursor = Cursors.Pen;
        else if (ToolEraser.IsChecked == true) ViewportCanvas.Cursor = Cursors.Cross;
        else ViewportCanvas.Cursor = Cursors.Hand;
    }

    private void Viewport_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (_sourceBitmap == null) return;

        var posInCanvas = e.GetPosition(ViewportCanvas);
        var posInImage = ViewportCanvasToImagePoint(posInCanvas);

        if (_isRulerActive)
        {
            if (_rulerStartPoint == null)
            {
                _rulerStartPoint = posInImage;
                RulerLine.X1 = posInImage.X;
                RulerLine.Y1 = posInImage.Y;
                RulerLine.X2 = posInImage.X;
                RulerLine.Y2 = posInImage.Y;
                RulerLine.Visibility = Visibility.Visible;
                RulerStatus.Text = "Clique no ponto final da barra de escala...";
            }
            else
            {
                FinishRulerCalibration(_rulerStartPoint.Value, posInImage);
            }
            return;
        }

        if (ToolPan.IsChecked == true)
        {
            _isPanning = true;
            _panStartMouse = posInCanvas;
            _panStartTranslate = new Point(ViewportTranslate.X, ViewportTranslate.Y);
            ViewportCanvas.CaptureMouse();
            ViewportCanvas.Cursor = Cursors.SizeAll;
        }
        else if (ToolBrush.IsChecked == true || ToolEraser.IsChecked == true)
        {
            PushUndoState();
            _isDrawing = true;
            ViewportCanvas.CaptureMouse();
            PaintOnMask(posInImage, ToolBrush.IsChecked == true);
        }
    }

    private void Viewport_MouseMove(object sender, MouseEventArgs e)
    {
        if (_sourceBitmap == null) return;

        var posInCanvas = e.GetPosition(ViewportCanvas);
        var posInImage = ViewportCanvasToImagePoint(posInCanvas);

        StatusCursorPos.Text = $"X: {(int)posInImage.X}, Y: {(int)posInImage.Y} px";

        if (_isPanning)
        {
            double dx = posInCanvas.X - _panStartMouse.X;
            double dy = posInCanvas.Y - _panStartMouse.Y;
            ViewportTranslate.X = _panStartTranslate.X + dx;
            ViewportTranslate.Y = _panStartTranslate.Y + dy;
            return;
        }

        if (_isRulerActive && _rulerStartPoint != null)
        {
            RulerLine.X2 = posInImage.X;
            RulerLine.Y2 = posInImage.Y;
            double len = Math.Sqrt(Math.Pow(posInImage.X - _rulerStartPoint.Value.X, 2) + Math.Pow(posInImage.Y - _rulerStartPoint.Value.Y, 2));
            RulerStatus.Text = $"Comprimento: {len:F1} pixels (clique para confirmar)";
            return;
        }

        if (_isDrawing && (ToolBrush.IsChecked == true || ToolEraser.IsChecked == true))
        {
            PaintOnMask(posInImage, ToolBrush.IsChecked == true);
        }
    }

    private void Viewport_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (_isPanning)
        {
            _isPanning = false;
            ViewportCanvas.ReleaseMouseCapture();
            UpdateCursor();
        }
        if (_isDrawing)
        {
            _isDrawing = false;
            ViewportCanvas.ReleaseMouseCapture();
        }
    }

    private Point ViewportCanvasToImagePoint(Point canvasPt)
    {
        double x = (canvasPt.X - ViewportTranslate.X) / ViewportScale.ScaleX;
        double y = (canvasPt.Y - ViewportTranslate.Y) / ViewportScale.ScaleY;
        return new Point(Math.Clamp(x, 0, _imgWidth - 1), Math.Clamp(y, 0, _imgHeight - 1));
    }

    private void PaintOnMask(Point pt, bool addMask)
    {
        if (_editableMaskBitmap == null) return;

        int cx = (int)pt.X;
        int cy = (int)pt.Y;
        int r = _brushRadius;
        int r2 = r * r;

        int minX = Math.Max(0, cx - r);
        int maxX = Math.Min(_imgWidth - 1, cx + r);
        int minY = Math.Max(0, cy - r);
        int maxY = Math.Min(_imgHeight - 1, cy + r);

        _editableMaskBitmap.Lock();
        unsafe
        {
            uint* pBuffer = (uint*)_editableMaskBitmap.BackBuffer;
            int stride = _editableMaskBitmap.BackBufferStride / 4;

            // Cor verde translúcida para a máscara: ARGB = (255, 34, 197, 94)
            uint colorVal = addMask ? 0xFF22C55E : 0x00000000;

            for (int y = minY; y <= maxY; y++)
            {
                for (int x = minX; x <= maxX; x++)
                {
                    if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2)
                    {
                        *(pBuffer + y * stride + x) = colorVal;
                    }
                }
            }
        }
        _editableMaskBitmap.AddDirtyRect(new Int32Rect(minX, minY, maxX - minX + 1, maxY - minY + 1));
        _editableMaskBitmap.Unlock();
    }

    private void PushUndoState()
    {
        if (_editableMaskBitmap == null) return;
        if (_undoStack.Count > 10)
        {
            var list = new System.Collections.Generic.List<byte[]>(_undoStack);
            list.RemoveAt(list.Count - 1);
            _undoStack.Clear();
            for (int i = list.Count - 1; i >= 0; i--) _undoStack.Push(list[i]);
        }
        _undoStack.Push(GetMaskBytesFromWriteableBitmap());
        if (BtnUndo != null) BtnUndo.IsEnabled = true;
    }

    private void UndoMask_Click(object sender, RoutedEventArgs e)
    {
        if (_undoStack.Count > 0 && _editableMaskBitmap != null)
        {
            var prev = _undoStack.Pop();
            LoadMaskBytesIntoWriteableBitmap(prev);
            StatusText.Text = "Último traço desfeito.";
            if (BtnUndo != null) BtnUndo.IsEnabled = _undoStack.Count > 0;
        }
    }

    private byte[] GetMaskBytesFromWriteableBitmap()
    {
        if (_editableMaskBitmap == null) return Array.Empty<byte>();
        byte[] grayMask = new byte[_imgWidth * _imgHeight];
        _editableMaskBitmap.Lock();
        unsafe
        {
            uint* pSrc = (uint*)_editableMaskBitmap.BackBuffer;
            int stride = _editableMaskBitmap.BackBufferStride / 4;
            for (int y = 0; y < _imgHeight; y++)
            {
                for (int x = 0; x < _imgWidth; x++)
                {
                    uint col = *(pSrc + y * stride + x);
                    grayMask[y * _imgWidth + x] = ((col >> 24) > 127) ? (byte)255 : (byte)0;
                }
            }
        }
        _editableMaskBitmap.Unlock();
        return grayMask;
    }

    private void RestoreOriginalMask_Click(object sender, RoutedEventArgs e)
    {
        if (_originalMaskBytes == null || _editableMaskBitmap == null)
        {
            ShowWarning("Nenhuma máscara automática original disponível para restauração.");
            return;
        }

        PushUndoState();
        LoadMaskBytesIntoWriteableBitmap(_originalMaskBytes);
        LogAudit("Máscara restaurada para o estado original do algoritmo automático.");
        StatusText.Text = "Máscara original restaurada.";
    }

    // ─────────────────────────────────────────────────────────────
    // 5. MOTOR CIENTÍFICO PYTHON & RECALCULO DE MÉTRICAS
    // ─────────────────────────────────────────────────────────────

    private async void RunAutoSegmentation_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_loadedImagePath))
        {
            ShowWarning("Selecione uma imagem de microscopia antes de executar.");
            return;
        }
        if (!ValidateCalibrationInput()) return;

        BtnRunAuto.IsEnabled = false;
        BtnRunAuto.Content = "⏳ Processando...";
        if (ProgressBarProcessing != null) ProgressBarProcessing.Visibility = Visibility.Visible;
        WarningBox.Visibility = Visibility.Collapsed;
        StatusText.Text = "Executando segmentação científica...";

        try
        {
            var repoDir = FindRepositoryRoot(AppContext.BaseDirectory);
            var outputDir = System.IO.Path.Combine(repoDir, "outputs");
            Directory.CreateDirectory(outputDir);

            var pythonExe = ResolvePythonExecutable(repoDir);
            var modTag = ((ComboBoxItem)CmbModule.SelectedItem).Tag?.ToString() ?? "fiber";
            var methodTag = ((ComboBoxItem)CmbMethod.SelectedItem).Tag?.ToString() ?? "otsu";

            var engineDir = System.IO.Path.Combine(repoDir, "engine");
            var startInfo = new ProcessStartInfo(pythonExe)
            {
                WorkingDirectory = repoDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            startInfo.Environment["PYTHONPATH"] = engineDir;

            startInfo.ArgumentList.Add("-m");
            startInfo.ArgumentList.Add("swiftmev.cli");
            startInfo.ArgumentList.Add("--module");
            startInfo.ArgumentList.Add(modTag);
            startInfo.ArgumentList.Add("--input");
            startInfo.ArgumentList.Add(_loadedImagePath);
            startInfo.ArgumentList.Add("--output");
            startInfo.ArgumentList.Add(outputDir);
            startInfo.ArgumentList.Add("--um-per-pixel");
            startInfo.ArgumentList.Add(_umPerPixel.ToString(CultureInfo.InvariantCulture));
            startInfo.ArgumentList.Add("--method");
            startInfo.ArgumentList.Add(methodTag);
            startInfo.ArgumentList.Add("--save-mask");

            if (ChkClahe.IsChecked == true) startInfo.ArgumentList.Add("--clahe");
            if (ChkInvert.IsChecked == true) startInfo.ArgumentList.Add("--invert");

            using var process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("Não foi possível iniciar o motor Python.");

            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            var stdout = await stdoutTask;
            var stderr = await stderrTask;

            if (process.ExitCode == 0)
            {
                HandleSegmentationSuccess(stdout, modTag);
            }
            else if (process.ExitCode == 2)
            {
                HandleValidationError(stderr);
            }
            else
            {
                ShowWarning($"Falha na execução do motor (Exit Code {process.ExitCode}):\n{stderr}");
            }
        }
        catch (Exception ex)
        {
            ShowWarning($"Erro: {ex.Message}");
        }
        finally
        {
            BtnRunAuto.IsEnabled = true;
            BtnRunAuto.Content = "⚡ Processar Imagem";
            if (ProgressBarProcessing != null) ProgressBarProcessing.Visibility = Visibility.Collapsed;
        }
    }

    private async void RecalculateMetrics_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_loadedImagePath) || _editableMaskBitmap == null)
        {
            ShowWarning("Nenhuma análise ativa para recálculo.");
            return;
        }
        if (!ValidateCalibrationInput()) return;

        BtnRecalculate.IsEnabled = false;
        BtnRecalculate.Content = "⏳ Recalculando...";
        StatusText.Text = "Recalculando métricas a partir da máscara editada...";

        try
        {
            var repoDir = FindRepositoryRoot(AppContext.BaseDirectory);
            var outputDir = System.IO.Path.Combine(repoDir, "outputs");
            Directory.CreateDirectory(outputDir);

            // Salvar máscara editada em PNG temporário
            var tempMaskPath = System.IO.Path.Combine(outputDir, "temp_edited_mask.png");
            SaveMaskToDisk(tempMaskPath);

            var pythonExe = ResolvePythonExecutable(repoDir);
            var modTag = ((ComboBoxItem)CmbModule.SelectedItem).Tag?.ToString() ?? "fiber";

            var engineDir = System.IO.Path.Combine(repoDir, "engine");
            var startInfo = new ProcessStartInfo(pythonExe)
            {
                WorkingDirectory = repoDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            startInfo.Environment["PYTHONPATH"] = engineDir;

            startInfo.ArgumentList.Add("-m");
            startInfo.ArgumentList.Add("swiftmev.cli");
            startInfo.ArgumentList.Add("--module");
            startInfo.ArgumentList.Add(modTag);
            startInfo.ArgumentList.Add("--input");
            startInfo.ArgumentList.Add(_loadedImagePath);
            startInfo.ArgumentList.Add("--mask");
            startInfo.ArgumentList.Add(tempMaskPath);
            startInfo.ArgumentList.Add("--output");
            startInfo.ArgumentList.Add(outputDir);
            startInfo.ArgumentList.Add("--um-per-pixel");
            startInfo.ArgumentList.Add(_umPerPixel.ToString(CultureInfo.InvariantCulture));

            using var process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("Não foi possível iniciar o motor Python.");

            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            var stdout = await stdoutTask;
            var stderr = await stderrTask;

            if (process.ExitCode == 0)
            {
                HandleSegmentationSuccess(stdout, modTag);
                LogAudit("Métricas recalculadas com sucesso a partir de alterações manuais do pesquisador.");
            }
            else
            {
                HandleValidationError(stderr);
            }
        }
        catch (Exception ex)
        {
            ShowWarning($"Erro ao recalcular: {ex.Message}");
        }
        finally
        {
            BtnRecalculate.IsEnabled = true;
            BtnRecalculate.Content = "📐 Recalcular Métricas da Máscara";
        }
    }

    private void HandleSegmentationSuccess(string stdout, string modTag)
    {
        try
        {
            using var doc = JsonDocument.Parse(stdout);
            _lastMetricsJson = doc.RootElement.Clone();
            var root = _lastMetricsJson;

            // Carregar Overlay (com fundo transparente para não ocultar a camada de máscara translúcida)
            if (root.TryGetProperty("overlay", out var overProp) && File.Exists(overProp.GetString()))
            {
                ImgOverlay.Source = CreateTransparentContourBitmap(overProp.GetString()!);
            }

            // Carregar Máscara Binária
            if (root.TryGetProperty("mask", out var maskProp) && !string.IsNullOrEmpty(maskProp.GetString()) && File.Exists(maskProp.GetString()))
            {
                _loadedMaskAutoPath = maskProp.GetString();
                var maskBytes = File.ReadAllBytes(_loadedMaskAutoPath!);
                _originalMaskBytes = maskBytes;
                LoadMaskFileIntoWriteableBitmap(_loadedMaskAutoPath!);
            }

            // Cards de Métricas
            if (root.TryGetProperty("metrics", out var metrics))
            {
                ValMetric1.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1565C0"));
                if (modTag == "fiber")
                {
                    LblMetric1Title.Text = "Diâmetro Mediano";
                    LblMetric2Title.Text = "Desvio Padrão";

                    if (metrics.TryGetProperty("median_diameter_um", out var med))
                        ValMetric1.Text = $"{med.GetDouble():F2} μm";
                    if (metrics.TryGetProperty("std_diameter_um", out var std))
                        ValMetric2.Text = $"± {std.GetDouble():F2} μm";

                    var meanUm = metrics.TryGetProperty("mean_diameter_um", out var mu) ? mu.GetDouble() : 0;
                    var meanPx = metrics.TryGetProperty("mean_diameter_px", out var mp) ? mp.GetDouble() : 0;
                    var medPx = metrics.TryGetProperty("median_diameter_px", out var medp) ? medp.GetDouble() : 0;
                    var pts = metrics.TryGetProperty("skeleton_points", out var sp) ? sp.GetInt32() : 0;

                    TxtStatsDetail.Text = $"• Média: {meanUm:F2} μm ({meanPx:F1} px)\n" +
                                          $"• Mediana: {ValMetric1.Text} ({medPx:F1} px)\n" +
                                          $"• Amostras no Esqueleto: {pts:N0} pontos\n" +
                                          $"• Calibração: {_umPerPixel:F4} μm/px";

                    if (metrics.TryGetProperty("resolution_warning", out var warn) && warn.ValueKind == JsonValueKind.True)
                    {
                        var msg = metrics.TryGetProperty("warning_message", out var wmsg) && wmsg.ValueKind == JsonValueKind.String ? wmsg.GetString() : "Alerta de discretização.";
                        ShowWarning(msg ?? "Resolução de imagem no limiar metrológico.");
                    }
                    else
                    {
                        WarningBox.Visibility = Visibility.Collapsed;
                    }
                }
                else
                {
                    WarningBox.Visibility = Visibility.Collapsed;
                    LblMetric1Title.Text = "Área Livre (Scratch)";
                    LblMetric2Title.Text = "Área Descoberta";

                    if (metrics.TryGetProperty("empty_area_fraction", out var frac))
                        ValMetric1.Text = $"{frac.GetDouble() * 100:F1} %";
                    if (metrics.TryGetProperty("empty_area_um2", out var aum))
                        ValMetric2.Text = $"{aum.GetDouble():N0} μm²";

                    var apx = metrics.TryGetProperty("empty_area_px2", out var ap) ? ap.GetInt32() : 0;
                    TxtStatsDetail.Text = $"• Fração Livre: {ValMetric1.Text}\n" +
                                          $"• Área Descoberta: {ValMetric2.Text}\n" +
                                          $"• Pixels Livres: {apx:N0} px²\n" +
                                          $"• Calibração: {_umPerPixel:F4} μm/px";
                }
            }

            // Latência e RSS
            if (root.TryGetProperty("latency", out var lat) && lat.TryGetProperty("wall_clock_time_ms", out var wall))
                ValLatency.Text = $"{wall.GetDouble():F1} ms";
            if (root.TryGetProperty("memory", out var mem) && mem.TryGetProperty("process_rss_mb", out var rss))
                ValMemory.Text = $"{rss.GetDouble():F1} MB";

            LogAudit($"Análise científica realizada com sucesso.\n{stdout}");
            StatusText.Text = "Processamento concluído com êxito.";
        }
        catch (Exception ex)
        {
            ShowWarning($"Erro ao interpretar resultados: {ex.Message}");
        }
    }

    private void HandleValidationError(string stderr)
    {
        string msg = stderr;
        try
        {
            using var doc = JsonDocument.Parse(stderr);
            if (doc.RootElement.TryGetProperty("message", out var mp)) msg = mp.GetString() ?? stderr;
        }
        catch { }

        ShowWarning($"Validação Científica Rejeitada: {msg}");
        ValMetric1.Text = "INVÁLIDO";
        ValMetric1.Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E5484D"));
        ValMetric2.Text = "--";
        StatusText.Text = "Operação bloqueada por critério metrológico.";
        LogAudit($"[FALHA DE VALIDAÇÃO]\n{stderr}");
    }

    private BitmapSource? CreateTransparentContourBitmap(string overlayPath)
    {
        try
        {
            var src = new BitmapImage();
            src.BeginInit();
            src.CacheOption = BitmapCacheOption.OnLoad;
            src.UriSource = new Uri(overlayPath, UriKind.Absolute);
            src.EndInit();

            var bgra = new FormatConvertedBitmap(src, PixelFormats.Bgra32, null, 0);
            int width = bgra.PixelWidth;
            int height = bgra.PixelHeight;
            int stride = width * 4;
            byte[] pixels = new byte[height * stride];
            bgra.CopyPixels(pixels, stride, 0);

            for (int i = 0; i < pixels.Length; i += 4)
            {
                byte b = pixels[i];
                byte g = pixels[i + 1];
                byte r = pixels[i + 2];

                // Os contornos gerados pelo OpenCV usam verde puro (B=0, G=255, R=0).
                // O fundo da imagem original convertida para 3 canais é puramente cinza (B == G == R).
                // Filtramos os pixels do contorno verde e tornamos o restante 100% transparente.
                if (g > 50 && (g - r) > 25 && (g - b) > 25)
                {
                    pixels[i] = 0;       // B
                    pixels[i + 1] = 255; // G
                    pixels[i + 2] = 0;   // R
                    pixels[i + 3] = 255; // A (opaco para o contorno verde)
                }
                else
                {
                    pixels[i] = 0;
                    pixels[i + 1] = 0;
                    pixels[i + 2] = 0;
                    pixels[i + 3] = 0;   // 100% transparente
                }
            }

            var result = BitmapSource.Create(width, height, 96, 96, PixelFormats.Bgra32, null, pixels, stride);
            result.Freeze();
            return result;
        }
        catch
        {
            return null;
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 6. MANIPULAÇÃO DE BYTES DA MÁSCARA & PERSISTÊNCIA
    // ─────────────────────────────────────────────────────────────

    private void LoadMaskFileIntoWriteableBitmap(string maskPath)
    {
        var maskBmp = new BitmapImage();
        maskBmp.BeginInit();
        maskBmp.CacheOption = BitmapCacheOption.OnLoad;
        maskBmp.UriSource = new Uri(maskPath, UriKind.Absolute);
        maskBmp.EndInit();

        var converted = new FormatConvertedBitmap(maskBmp, PixelFormats.Gray8, null, 0);
        int stride = _imgWidth;
        byte[] pixels = new byte[_imgWidth * _imgHeight];
        converted.CopyPixels(pixels, stride, 0);

        LoadMaskBytesIntoWriteableBitmap(pixels);
    }

    private void LoadMaskBytesIntoWriteableBitmap(byte[] grayPixels)
    {
        if (_editableMaskBitmap == null) return;

        _editableMaskBitmap.Lock();
        unsafe
        {
            uint* pDest = (uint*)_editableMaskBitmap.BackBuffer;
            int stride = _editableMaskBitmap.BackBufferStride / 4;

            for (int y = 0; y < _imgHeight; y++)
            {
                for (int x = 0; x < _imgWidth; x++)
                {
                    byte val = grayPixels[y * _imgWidth + x];
                    *(pDest + y * stride + x) = (val > 127) ? 0xFF22C55E : 0x00000000;
                }
            }
        }
        _editableMaskBitmap.AddDirtyRect(new Int32Rect(0, 0, _imgWidth, _imgHeight));
        _editableMaskBitmap.Unlock();
    }

    private void SaveMaskToDisk(string targetPngPath)
    {
        if (_editableMaskBitmap == null) return;

        byte[] grayMask = new byte[_imgWidth * _imgHeight];
        _editableMaskBitmap.Lock();
        unsafe
        {
            uint* pSrc = (uint*)_editableMaskBitmap.BackBuffer;
            int stride = _editableMaskBitmap.BackBufferStride / 4;

            for (int y = 0; y < _imgHeight; y++)
            {
                for (int x = 0; x < _imgWidth; x++)
                {
                    uint col = *(pSrc + y * stride + x);
                    grayMask[y * _imgWidth + x] = ((col >> 24) > 127) ? (byte)255 : (byte)0;
                }
            }
        }
        _editableMaskBitmap.Unlock();

        var grayBitmap = BitmapSource.Create(_imgWidth, _imgHeight, 96, 96, PixelFormats.Gray8, null, grayMask, _imgWidth);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(grayBitmap));
        using var stream = File.Create(targetPngPath);
        encoder.Save(stream);
    }

    // ─────────────────────────────────────────────────────────────
    // 7. PROJETOS PERSISTENTES & EXPORTAÇÃO CIENTÍFICA
    // ─────────────────────────────────────────────────────────────

    private void SaveProject_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_loadedImagePath))
        {
            ShowWarning("Nenhuma análise ativa para salvar como projeto.");
            return;
        }

        var dialog = new SaveFileDialog
        {
            Title = "Salvar Projeto SwiftMev",
            Filter = "Projeto SwiftMev (*.json)|*.json",
            FileName = $"{System.IO.Path.GetFileNameWithoutExtension(_loadedImagePath)}_analysis.json"
        };

        if (dialog.ShowDialog() == true)
        {
            try
            {
                var projectDir = System.IO.Path.GetDirectoryName(dialog.FileName)!;
                var baseName = System.IO.Path.GetFileNameWithoutExtension(dialog.FileName);

                var maskPath = System.IO.Path.Combine(projectDir, $"{baseName}_mask.png");
                SaveMaskToDisk(maskPath);

                var projectData = new
                {
                    application = "SwiftMev",
                    version = "0.4.0",
                    created_at_utc = DateTime.UtcNow.ToString("o"),
                    source_image_path = _loadedImagePath,
                    calibration_um_per_pixel = _umPerPixel,
                    module = ((ComboBoxItem)CmbModule.SelectedItem).Tag?.ToString(),
                    mask_file = maskPath,
                    metrics = _lastMetricsJson.ValueKind != JsonValueKind.Undefined ? _lastMetricsJson : default
                };

                File.WriteAllText(dialog.FileName, JsonSerializer.Serialize(projectData, new JsonSerializerOptions { WriteIndented = true }));
                StatusText.Text = "Projeto salvo com sucesso.";
                LogAudit($"Projeto persistido em: {dialog.FileName}");
            }
            catch (Exception ex)
            {
                ShowWarning($"Falha ao salvar projeto: {ex.Message}");
            }
        }
    }

    private void OpenProject_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "Abrir Projeto SwiftMev",
            Filter = "Projeto SwiftMev (*.json)|*.json"
        };

        if (dialog.ShowDialog() == true)
        {
            try
            {
                var jsonText = File.ReadAllText(dialog.FileName);
                using var doc = JsonDocument.Parse(jsonText);
                var root = doc.RootElement;

                var imgPath = root.GetProperty("source_image_path").GetString();
                if (!string.IsNullOrEmpty(imgPath) && File.Exists(imgPath))
                {
                    LoadImageFile(imgPath);
                }

                if (root.TryGetProperty("calibration_um_per_pixel", out var calProp))
                {
                    _umPerPixel = calProp.GetDouble();
                    TxtCalibration.Text = _umPerPixel.ToString(CultureInfo.InvariantCulture);
                }

                if (root.TryGetProperty("mask_file", out var maskProp) && File.Exists(maskProp.GetString()))
                {
                    LoadMaskFileIntoWriteableBitmap(maskProp.GetString()!);
                }

                LogAudit($"Projeto restaurado de: {dialog.FileName}");
                StatusText.Text = "Projeto reaberto com sucesso.";
            }
            catch (Exception ex)
            {
                ShowWarning($"Falha ao abrir projeto: {ex.Message}");
            }
        }
    }

    private void ExportReport_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_loadedImagePath))
        {
            ShowWarning("Carregue e processe uma imagem antes de exportar.");
            return;
        }

        var dialog = new SaveFileDialog
        {
            Title = "Exportar Relatório e Anotação",
            Filter = "Relatório CSV (*.csv)|*.csv",
            FileName = $"{System.IO.Path.GetFileNameWithoutExtension(_loadedImagePath)}_relatorio.csv"
        };

        if (dialog.ShowDialog() == true)
        {
            try
            {
                var csv = new StringBuilder();
                csv.AppendLine("Parametro,Valor,Unidade");
                csv.AppendLine($"Arquivo_Origem,{_loadedImagePath},caminho");
                csv.AppendLine($"Calibracao,{_umPerPixel},um/pixel");
                csv.AppendLine($"Metrica_Primaria,{ValMetric1.Text},--");
                csv.AppendLine($"Metrica_Secundaria,{ValMetric2.Text},--");
                csv.AppendLine($"Data_UTC,{DateTime.UtcNow:o},ISO8601");
                csv.AppendLine($"Software,SwiftMev Alpha 0.4,SoftwareX");

                File.WriteAllText(dialog.FileName, csv.ToString(), Encoding.UTF8);

                // Exportar imagem anotada combinada
                var imgOut = System.IO.Path.ChangeExtension(dialog.FileName, ".png");
                ExportCombinedCanvasToImage(imgOut);

                StatusText.Text = "Relatório exportado com sucesso.";
                LogAudit($"Exportação científica concluída:\n• CSV: {dialog.FileName}\n• Imagem: {imgOut}");
            }
            catch (Exception ex)
            {
                ShowWarning($"Falha ao exportar relatório: {ex.Message}");
            }
        }
    }

    private void ExportCombinedCanvasToImage(string outputPath)
    {
        if (_sourceBitmap == null) return;

        var rtb = new RenderTargetBitmap(_imgWidth, _imgHeight, 96, 96, PixelFormats.Pbgra32);
        var dv = new DrawingVisual();
        using (var dc = dv.RenderOpen())
        {
            // Imagem original
            dc.DrawImage(_sourceBitmap, new Rect(0, 0, _imgWidth, _imgHeight));

            // Máscara
            if (_editableMaskBitmap != null && ChkShowMask.IsChecked == true)
            {
                dc.PushOpacity(SliderMaskAlpha.Value);
                dc.DrawImage(_editableMaskBitmap, new Rect(0, 0, _imgWidth, _imgHeight));
                dc.Pop();
            }

            // Contornos
            if (ImgOverlay.Source != null && ChkShowOverlay.IsChecked == true)
            {
                dc.DrawImage(ImgOverlay.Source, new Rect(0, 0, _imgWidth, _imgHeight));
            }

            // Barra de escala gravada no canto
            if (ChkShowScaleBar.IsChecked == true && _umPerPixel > 0)
            {
                double barUm = RoundToNiceScale(100 * _umPerPixel);
                double barPx = barUm / _umPerPixel;
                var rect = new Rect(_imgWidth - barPx - 30, _imgHeight - 40, barPx, 6);
                dc.DrawRectangle(Brushes.White, null, rect);

                var ft = new FormattedText(
                    $"{barUm} μm",
                    CultureInfo.InvariantCulture,
                    FlowDirection.LeftToRight,
                    new Typeface("Arial"),
                    14,
                    Brushes.White,
                    96
                );
                dc.DrawText(ft, new Point(_imgWidth - barPx - 30 + (barPx - ft.Width) / 2, _imgHeight - 65));
            }
        }
        rtb.Render(dv);

        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(rtb));
        using var stream = File.Create(outputPath);
        encoder.Save(stream);
    }

    // ─────────────────────────────────────────────────────────────
    // 8. HELPERS, CONTROLES DE CAMADAS & VALIDAÇÃO
    // ─────────────────────────────────────────────────────────────

    private void LayerVisibility_Changed(object sender, RoutedEventArgs e)
    {
        if (!IsLoaded || ImgOverlay == null || ImgMaskLayer == null) return;
        ImgOverlay.Visibility = (ChkShowOverlay?.IsChecked == true) ? Visibility.Visible : Visibility.Collapsed;
        ImgMaskLayer.Visibility = (ChkShowMask?.IsChecked == true) ? Visibility.Visible : Visibility.Collapsed;
        UpdateScaleBar();
    }

    private void SliderMaskAlpha_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (ImgMaskLayer != null) ImgMaskLayer.Opacity = e.NewValue;
    }

    private bool ValidateCalibrationInput()
    {
        var text = TxtCalibration.Text.Trim().Replace(',', '.');
        if (string.IsNullOrWhiteSpace(text) || !_isCalibrated || !double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var cal) || cal <= 0 || double.IsNaN(cal))
        {
            ShowWarning("Calibração espacial pendente! Para garantir rigor metrológico (SoftwareX), defina a escala física na aba 'Calibração' antes de processar.");
            if (TabCalibration != null) TabCalibration.IsSelected = true;
            return false;
        }
        _umPerPixel = cal;
        return true;
    }

    private void ShowWarning(string message)
    {
        TxtWarningMessage.Text = message;
        WarningBox.Visibility = Visibility.Visible;
    }

    private void LogAudit(string message)
    {
        var time = DateTime.Now.ToString("HH:mm:ss");
        TxtAuditLog.AppendText($"[{time}] {message}\n\n");
        TxtAuditLog.ScrollToEnd();
    }

    private static string FindRepositoryRoot(string currentDir)
    {
        var dir = new DirectoryInfo(currentDir);
        while (dir != null)
        {
            if (Directory.Exists(System.IO.Path.Combine(dir.FullName, "engine", "swiftmev")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return System.IO.Path.GetFullPath(System.IO.Path.Combine(currentDir, "../../../../"));
    }

    private static string ResolvePythonExecutable(string repoDir)
    {
        var envPython = Environment.GetEnvironmentVariable("SWIFTMEV_PYTHON")
            ?? Environment.GetEnvironmentVariable("CELLSCOPE_PYTHON");
        if (!string.IsNullOrEmpty(envPython) && File.Exists(envPython))
            return envPython;

        var venvPython = System.IO.Path.Combine(repoDir, "engine", ".venv", "Scripts", "python.exe");
        if (File.Exists(venvPython))
            return venvPython;

        return "python";
    }
}
