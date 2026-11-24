"""Testes automatizados do PixelLab.
Execute com: python -m unittest -v test_pixellab_responsivo.py
"""
import os
import unittest

import cv2
import numpy as np

import pixellab_app_responsivo as app


class PixelLabTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((80, 100, 3), dtype=np.uint8)
        self.image[10:65, 12:60] = [220, 70, 45]
        self.image[30:72, 60:92] = [20, 180, 220]

    def assert_rgb_image(self, image):
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)

    def test_adjustment_operations(self):
        modes = [
            "Brilho e contraste", "Correção de gama", "Negativo", "Escala de cinza",
            "Equalização do histograma", "Canal vermelho", "Canal verde", "Canal azul",
        ]
        modes.append("Saturação")
        for mode in modes:
            with self.subTest(mode=mode):
                result, intermediate, label, explanation = app.apply_adjustment(self.image, mode)
                self.assert_rgb_image(result)
                self.assert_rgb_image(intermediate)
                self.assertTrue(label)
                self.assertTrue(explanation)

    def test_filter_operations(self):
        for mode in ["Gaussiano", "Mediana", "Bilateral", "Nitidez (sharpen)", "Sobel", "Canny", "Laplaciano"]:
            with self.subTest(mode=mode):
                result, intermediate, label, explanation = app.apply_filter(self.image, mode)
                self.assert_rgb_image(result)
                self.assert_rgb_image(intermediate)
                self.assertTrue(label)
                self.assertTrue(explanation)

    def test_segmentation_operations(self):
        for mode in ["Limiarização fixa", "Otsu", "Adaptativa", "Cor (HSV)"]:
            with self.subTest(mode=mode):
                result, intermediate, label, explanation = app.apply_segmentation(self.image, mode)
                self.assert_rgb_image(result)
                self.assert_rgb_image(intermediate)
                self.assertTrue(label)
                self.assertTrue(explanation)

    def test_transform_returns_rgb_and_changes_canvas(self):
        result, intermediate, label, explanation = app.apply_transform(self.image, angle=45, scale=1.0)
        self.assert_rgb_image(result)
        self.assert_rgb_image(intermediate)
        self.assertNotEqual(result.shape[:2], self.image.shape[:2])
        self.assertTrue(label)
        self.assertTrue(explanation)

    def test_gamma_below_one_brightens_midtones(self):
        midtone = np.full((8, 8, 3), 64, dtype=np.uint8)
        result, *_ = app.apply_adjustment(midtone, "Correção de gama", gamma=0.5)
        self.assertGreater(int(result[0, 0, 0]), 64)

    def test_normalize_grayscale_rgba_and_large_images(self):
        gray = np.full((10, 12), 128, dtype=np.uint8)
        self.assert_rgb_image(app.normalize_image(gray))
        rgba = np.zeros((10, 12, 4), dtype=np.uint8)
        rgba[:, :, 0] = 200
        rgb = app.normalize_image(rgba)
        self.assert_rgb_image(rgb)
        self.assertEqual(int(rgb[0, 0, 0]), 200)
        large = np.zeros((40, 80, 3), dtype=np.uint8)
        resized = app.normalize_image(large, max_side=20)
        self.assertEqual(max(resized.shape[:2]), 20)

    def test_load_image_initializes_original_and_current(self):
        loaded = app.load_image(self.image)
        self.assertTrue(np.array_equal(loaded[0], self.image))
        self.assertTrue(np.array_equal(loaded[1], self.image))
        self.assertEqual(loaded[2], [])
        self.assertEqual(loaded[3], [])
        self.assertIn("Imagem carregada", loaded[4])

    def test_no_image_prevents_processing_with_clear_status(self):
        callback = app.make_process_callback("filter")
        output = callback(None, None, [], [], [], "Canny", 5, 2.0, 60, 150)
        self.assertIsNone(output[0])
        self.assertIsNone(output[4])
        self.assertIn("Nenhuma imagem", output[9])

    def test_consecutive_operations_undo_redo_and_restore(self):
        loaded = app.load_image(self.image)
        process_adjust = app.make_process_callback("adjust")
        process_filter = app.make_process_callback("filter")

        first = process_adjust(*loaded[:5], "Negativo", 0, 1.0, 1.0)
        negative = first[4].copy()
        self.assertFalse(np.array_equal(negative, self.image))

        second = process_filter(loaded[0], first[4], first[5], first[6], first[7], "Canny", 5, 2.0, 50, 120)
        canny = second[4].copy()
        self.assertFalse(np.array_equal(canny, negative))
        self.assertGreaterEqual(len(second[5]), 2)
        self.assertEqual(len(second[6]), 0)

        undone = app.undo_operation(loaded[0], second[4], second[5], second[6], second[7])
        self.assertTrue(np.array_equal(undone[0], negative))
        self.assertEqual(len(undone[5]), 1)
        self.assertEqual(len(undone[6]), 1)

        redone = app.redo_operation(loaded[0], undone[4], undone[5], undone[6], undone[7])
        self.assertTrue(np.array_equal(redone[0], canny))
        restored = app.restore_original(loaded[0], redone[4], redone[5], redone[6], redone[7])
        self.assertTrue(np.array_equal(restored[0], self.image))
        self.assertTrue(np.array_equal(restored[4], self.image))

    def test_restore_without_image_and_undo_without_history(self):
        restored = app.restore_original(None, None, [], [], [])
        self.assertIsNone(restored[0])
        self.assertIn("Nenhuma imagem", restored[9])
        undone = app.undo_operation(None, None, [], [], [])
        self.assertIsNone(undone[0])
        self.assertIn("histórico", undone[9].lower())

    def test_ui_category_switch_does_not_mutate_image_state(self):
        original = self.image.copy()
        expected_panels = {
            "🎚️ Ajustes": 0,
            "◌ Filtros": 1,
            "↔ Transformações": 2,
            "◉ Segmentação e detecção": 3,
        }
        for category, expected_index in expected_panels.items():
            visibility = app.select_tool(category)
            self.assertEqual(len(visibility), 4)
            visible = [item.get("visible") for item in visibility]
            self.assertEqual(visible.count(True), 1)
            self.assertTrue(visible[expected_index])
            self.assertTrue(np.array_equal(original, self.image))

    def test_dynamic_control_visibility(self):
        linear = app.adjustment_controls_visibility("Brilho e contraste")
        gamma = app.adjustment_controls_visibility("Correção de gama")
        saturation = app.adjustment_controls_visibility("Saturação")
        self.assertEqual([item.get("visible") for item in linear], [True, True, False, False])
        self.assertEqual([item.get("visible") for item in gamma], [False, False, True, False])
        self.assertEqual([item.get("visible") for item in saturation], [False, False, False, True])

        gaussian = app.filter_controls_visibility("Gaussiano")
        sharpen = app.filter_controls_visibility("Nitidez (sharpen)")
        self.assertEqual([item.get("visible") for item in gaussian], [True, True])
        self.assertEqual([item.get("visible") for item in sharpen], [False, False])

        hsv = app.segmentation_controls_visibility("Cor (HSV)")
        otsu = app.segmentation_controls_visibility("Otsu")
        canny = app.segmentation_controls_visibility("Canny")
        self.assertEqual([item.get("visible") for item in hsv], [False, False, True, False])
        self.assertEqual([item.get("visible") for item in otsu], [False, False, False, False])
        self.assertEqual([item.get("visible") for item in canny], [False, False, False, True])
        self.assertIn("Canny", str(app.demo.get_config_file().get("components", [])))

    def test_build_app_contains_three_panel_layout_and_collapsible_analysis(self):
        config = app.demo.get_config_file()
        components = config.get("components", [])
        elem_ids = {component.get("props", {}).get("elem_id") for component in components}
        self.assertIn("pl-workspace", elem_ids)
        self.assertIn("pl-nav-panel", elem_ids)
        self.assertIn("pl-canvas-column", elem_ids)
        self.assertIn("pl-tools-panel", elem_ids)
        self.assertIn("pl-tool-nav", elem_ids)
        self.assertTrue(config.get("fill_width"))
        self.assertTrue(config.get("fill_height"))
        image_components = {
            component.get("props", {}).get("elem_id"): component.get("props", {}).get("height")
            for component in components
            if component.get("props", {}).get("elem_id") in {"pl-original-image", "pl-result-image"}
        }
        self.assertEqual(image_components, {"pl-original-image": 560, "pl-result-image": 560})
        self.assertIn("max-width: none", app.CSS)
        self.assertIn("grid-template-columns:clamp(150px, 11vw, 184px) minmax(0, 1fr) clamp(285px, 22vw, 350px)", app.CSS)
        self.assertIn("height:clamp(340px, 56vh, 620px)", app.CSS)
        self.assertIn("@media (max-width: 1250px) and (min-width: 851px)", app.CSS)
        self.assertNotIn("height:calc(100vh - 112px)", app.CSS)
        self.assertIn("pl-empty-state", elem_ids)
        self.assertIn("pl-adjust-panel", elem_ids)
        self.assertIn("pl-filter-panel", elem_ids)
        self.assertIn("pl-transform-panel", elem_ids)
        self.assertIn("pl-segment-panel", elem_ids)
        # The Gradio graph declares the category switching and parameter-change events.
        self.assertGreaterEqual(len(config.get("dependencies", [])), 10)

    def test_navigation_and_inspector_are_separate_panels(self):
        config = app.demo.get_config_file()
        components = config.get("components", [])
        component_ids = {
            component.get("props", {}).get("elem_id")
            for component in components
        }
        self.assertTrue({"pl-nav-panel", "pl-canvas-column", "pl-tools-panel"}.issubset(component_ids))
        nav_component = next(c for c in components if c.get("props", {}).get("elem_id") == "pl-nav-panel")
        inspector_component = next(c for c in components if c.get("props", {}).get("elem_id") == "pl-tools-panel")
        self.assertEqual(nav_component.get("type"), "column")
        self.assertEqual(inspector_component.get("type"), "column")

    def test_inspector_uses_visible_tool_lists_without_dropdowns(self):
        config = app.demo.get_config_file()
        components = config.get("components", [])
        dropdowns = [component for component in components if component.get("type") == "dropdown"]
        self.assertEqual(dropdowns, [], "O inspetor não deve conter menus suspensos.")

        expected_lists = {
            "pl-adjust-tool-list": "Brilho e contraste",
            "pl-filter-tool-list": "Gaussiano",
            "pl-flip-tool-list": "Nenhum",
            "pl-segment-tool-list": "Otsu",
        }
        by_id = {
            component.get("props", {}).get("elem_id"): component
            for component in components
        }
        for elem_id, expected_value in expected_lists.items():
            self.assertIn(elem_id, by_id)
            self.assertEqual(by_id[elem_id].get("type"), "radio")
            self.assertEqual(by_id[elem_id].get("props", {}).get("value"), expected_value)

        self.assertIn("pl-tool-picker fieldset > label", app.CSS)
        self.assertIn("pl-flip-tool-list fieldset", app.CSS)


    def test_saturation_zero_removes_chroma(self):
        result, *_ = app.apply_adjustment(self.image, "Saturação", saturation=0.0)
        self.assertTrue(np.array_equal(result[:, :, 0], result[:, :, 1]))
        self.assertTrue(np.array_equal(result[:, :, 1], result[:, :, 2]))

    def test_live_preview_is_based_on_confirmed_current_image(self):
        preview = app.make_preview_callback("adjust")
        first = preview(self.image, self.image, "Brilho e contraste", 20, 1.0, 1.0, 1.0)
        second = preview(self.image, self.image, "Brilho e contraste", 60, 1.0, 1.0, 1.0)
        expected = np.clip(self.image.astype(np.float32) + 60, 0, 255).astype(np.uint8)
        self.assertTrue(np.array_equal(second[0], expected))
        self.assertFalse(np.array_equal(first[0], second[0]))
        self.assertEqual(second[4]["label"], "Brilho e contraste (brilho=+60, contraste=1.00)")

    def test_confirm_preview_commits_and_cancel_discards(self):
        loaded = app.load_image(self.image)
        preview = app.make_preview_callback("segment")(*[loaded[0], loaded[1], "Canny", 127, 21, 0, 179, 50, 50, 30, 100])
        preview_payload = preview[4]
        cancelled = app.cancel_preview(loaded[0], loaded[1], loaded[2], loaded[3], loaded[4])
        self.assertTrue(np.array_equal(cancelled[0], self.image))
        self.assertTrue(np.array_equal(cancelled[4], self.image))
        self.assertIsNone(cancelled[10])

        confirmed = app.confirm_preview(loaded[0], loaded[1], loaded[2], loaded[3], loaded[4], preview_payload)
        self.assertTrue(np.array_equal(confirmed[4], preview_payload["result"]))
        self.assertEqual(len(confirmed[5]), 1)
        self.assertEqual(len(confirmed[6]), 0)
        self.assertIn(preview_payload["label"], confirmed[7])
        self.assertIsNone(confirmed[10])

    def test_hsv_preview_returns_masked_rgb_and_pending_payload(self):
        preview = app.make_preview_callback("segment")
        result = preview(self.image, self.image, "Cor (HSV)", 127, 21, 0, 179, 60, 60, 60, 150)
        self.assert_rgb_image(result[0])
        self.assertIsInstance(result[4], dict)
        self.assertIn("Segmentação por cor HSV", result[4]["label"])
        canny = preview(self.image, self.image, "Canny", 127, 21, 0, 179, 50, 50, 30, 100)
        self.assert_rgb_image(canny[0])
        self.assertIn("Canny", canny[4]["label"])

    def test_ui_has_debounced_preview_events_and_empty_workspace(self):
        config = app.demo.get_config_file()
        components = config.get("components", [])
        elem_ids = {component.get("props", {}).get("elem_id") for component in components}
        self.assertIn("pl-empty-state", elem_ids)
        preview_dependencies = [dep for dep in config.get("dependencies", []) if dep.get("js") and "setTimeout" in dep["js"]]
        cancellation_dependencies = [dep for dep in config.get("dependencies", []) if dep.get("js") and "clearTimeout" in dep["js"]]
        self.assertGreaterEqual(len(preview_dependencies), 10)
        self.assertGreaterEqual(len(cancellation_dependencies), 4)
        self.assertTrue(all("150" in dep["js"] for dep in preview_dependencies))

    def test_export_ui_only_reveals_download_when_png_exists(self):
        no_path, no_status = app.export_image_for_ui(None)
        self.assertFalse(no_path.get("visible"))
        self.assertIn("Carregue", no_status)
        download, status = app.export_image_for_ui(self.image)
        try:
            self.assertTrue(download.get("visible"))
            self.assertTrue(os.path.isfile(download.get("value")))
            self.assertIn("exportado", status)
        finally:
            path = download.get("value")
            if path and os.path.isfile(path):
                os.remove(path)

    def test_export_writes_valid_png(self):
        path, status = app.export_image(self.image)
        try:
            self.assertTrue(path)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)
            decoded = cv2.imread(path, cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape[:2], self.image.shape[:2])
            self.assertIn("exportado", status)
        finally:
            if path and os.path.isfile(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
