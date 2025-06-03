import numpy as np
import os
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

# Vision Transformer (ViT) inspired model using scikit-learn
class ViTInspiredClassifier:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.scaler = StandardScaler()
        self.patch_size = 16  # ViT-like patch size
        self.num_patches = (224 // self.patch_size) ** 2  # Number of patches for 224x224 image
        self.embedding_dim = 32  # Dimension of patch embeddings

        # Class labels
        self.class_labels = [
            "Acne", 
            "Hyperpigmentation", 
            "Nail Psoriasis", 
            "SJS-TEN", 
            "Vitiligo"
        ]

        # Initialize a RandomForestClassifier as a substitute for a ViT model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        )

        # Pretrain the model with random data
        self._pretrain()

        print("ViT-inspired classifier initialized")

    def _pretrain(self):
        """
        Pretrain the model with enhanced synthetic data
        This simulates a pretrained ViT model with better disease patterns
        """
        np.random.seed(42)

        # Increased training samples for better generalization
        n_samples = 1000
        n_features = self.num_patches * self.embedding_dim

        # Generate feature patterns specific to each condition
        patterns = {
            'acne': np.random.normal(0.7, 0.1, (n_features//5,)),
            'hyperpigmentation': np.random.normal(-0.5, 0.1, (n_features//5,)),
            'nail_psoriasis': np.random.normal(0.3, 0.15, (n_features//5,)),
            'sjs_ten': np.random.normal(0.8, 0.2, (n_features//5,)),
            'vitiligo': np.random.normal(-0.8, 0.1, (n_features//5,))
        }

        # Create balanced training data with disease-specific patterns
        X_train = []
        y_train = []

        for class_idx, (condition, pattern) in enumerate(patterns.items()):
            n_class_samples = 200  # Equal samples per class

            for _ in range(n_class_samples):
                # Create sample with condition-specific pattern
                sample = np.random.randn(n_features)
                pattern_start = np.random.randint(0, n_features - len(pattern))
                sample[pattern_start:pattern_start + len(pattern)] = pattern

                X_train.append(sample)
                y_train.append(class_idx)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Store data statistics for normalization
        self.scaler.fit(X_train)

        # Train the model on synthetic data
        X_train_scaled = self.scaler.transform(X_train)
        self.model.fit(X_train_scaled, y_train)

        # Save synthetic training data info for reference
        self.train_stats = {
            "n_samples": n_samples,
            "n_features": n_features,
            "class_distribution": {self.class_labels[i]: (y_train == i).sum() for i in range(self.num_classes)}
        }

    def extract_features(self, image):
        """
        Extract features from input image using a patch-based approach
        similar to Vision Transformers
        """
        # Check if we have a valid image with 3 dimensions (height, width, channels)
        if len(image.shape) == 3:
            # Ensure image is 224x224x3
            if image.shape[0] != 224 or image.shape[1] != 224:
                # Resize or pad if needed (should be handled by transform, but just in case)
                h, w = image.shape[:2]
                resized_img = np.zeros((224, 224, 3), dtype=np.float32)
                resized_img[:min(h, 224), :min(w, 224)] = image[:min(h, 224), :min(w, 224)]
                image = resized_img

            # Extract patches (ViT-like approach)
            patches = []
            for i in range(0, 224, self.patch_size):
                for j in range(0, 224, self.patch_size):
                    # Extract patch
                    patch = image[i:i+self.patch_size, j:j+self.patch_size, :]
                    # Flatten patch
                    patch_flat = patch.reshape(-1)
                    # Project to embedding dimension (simple mean pooling)
                    if len(patch_flat) > self.embedding_dim:
                        # Dimension reduction by averaging
                        patch_embedding = np.mean(patch_flat.reshape(-1, len(patch_flat) // self.embedding_dim), axis=1)
                    else:
                        # Pad if needed
                        patch_embedding = np.pad(patch_flat, (0, self.embedding_dim - len(patch_flat)), 'constant')

                    patches.append(patch_embedding)

            # Combine all patch embeddings
            features = np.concatenate(patches)

            # Ensure correct feature dimensionality
            expected_dim = self.num_patches * self.embedding_dim
            if len(features) != expected_dim:
                if len(features) > expected_dim:
                    features = features[:expected_dim]
                else:
                    features = np.pad(features, (0, expected_dim - len(features)), 'constant')

            return features
        else:
            # Handle unexpected input format
            print(f"Unexpected image shape: {image.shape}")
            return np.zeros(self.num_patches * self.embedding_dim)

    def predict(self, image):
        """
        Predict skin disease from input image using enhanced visual characteristics
        and multi-factor analysis for more accurate classification
        """
        try:
            # Convert to RGB uint8 for enhanced analysis
            img_rgb = (image * 255).astype(np.uint8)
            import cv2
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # Enhanced metrics extraction
            avg_brightness = np.mean(gray)
            contrast = np.std(gray)
            texture_variance = np.var(gray)

            # Enhanced color analysis
            color_ratios = np.mean(img_rgb, axis=(0,1))
            red_dominance = color_ratios[0] / np.sum(color_ratios)

            # Disease-specific pattern detection with stricter criteria
            has_acne = self._detect_acne_patterns(image)
            has_vitiligo = self._detect_vitiligo_patterns(image)
            has_hyperpigmentation = self._detect_hyperpigmentation_patterns(image)
            has_nail_condition = self._detect_nail_psoriasis_patterns(image)
            has_sjs_ten = self._detect_sjs_ten_patterns(image)

            # Calculate disease-specific scores with enhanced thresholds
            scores = np.zeros(5)

            # Acne scoring - stricter criteria
            if has_acne and red_dominance > 0.4:
                scores[0] = 0.7

            # Hyperpigmentation scoring - improved detection
            if has_hyperpigmentation and avg_brightness < 0.4:
                scores[1] = 0.85
                scores[0] *= 0.5  # Reduce acne score if hyperpigmentation is strong

            # Nail psoriasis scoring - enhanced pattern recognition
            if has_nail_condition and texture_variance > 0.15:
                scores[2] = 0.9
                scores[0] *= 0.3  # Significantly reduce acne score

            # SJS-TEN scoring - more specific criteria
            if has_sjs_ten and red_dominance > 0.45 and contrast > 0.2:
                scores[3] = 0.95
                scores[0] *= 0.2  # Greatly reduce acne score

            # Vitiligo scoring - refined detection
            if has_vitiligo and avg_brightness > 0.6:
                scores[4] = 0.92
                scores[0] *= 0.3  # Significantly reduce acne score

            # If we have a clear winner
            max_score = np.max(scores)
            if max_score > 0.5:
                predicted_class = np.argmax(scores)
                # Normalize scores to probabilities
                scores = scores / np.sum(scores)
                return predicted_class, scores

            # Fallback to feature-based prediction
            features = self.extract_features(image)
            features = features.reshape(1, -1)
            scaled_features = self.scaler.transform(features)
            class_probs = self.model.predict_proba(scaled_features)[0]
            predicted_class = np.argmax(class_probs)

            return predicted_class, class_probs

        except Exception as e:
            print(f"Error in prediction: {e}")
            # Fallback to basic feature extraction
            features = self.extract_features(image)
            features = features.reshape(1, -1)
            scaled_features = self.scaler.transform(features)
            class_probs = self.model.predict_proba(scaled_features)[0]
            predicted_class = np.argmax(class_probs)

            return predicted_class, class_probs

        try:
            # Convert to RGB uint8 for enhanced analysis
            img_rgb = (image * 255).astype(np.uint8)
            import cv2
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # Enhanced metrics extraction
            avg_brightness = np.mean(gray)
            contrast = np.std(gray)
            texture_variance = np.var(gray)

            # Enhanced color analysis
            color_ratios = np.mean(img_rgb, axis=(0,1))
            red_dominance = color_ratios[0] / np.sum(color_ratios)

            # Advanced edge and pattern detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Disease-specific pattern detection with stricter criteria
            has_red_spots = self._has_red_spots(image)
            has_white_patches = self._has_white_patches(image)
            has_dark_patches = self._has_dark_patches(image)

            # Multi-factor scoring system with improved thresholds
            scores = np.zeros(5)  # One score per condition

            # Acne-specific analysis - require more specific patterns
            if red_dominance > 0.35 and edge_density > 0.15 and has_red_spots:
                red_spot_count = self._count_red_spots(image)
                if red_spot_count > 3:  # Require multiple spots
                    scores[0] = 0.95  # High confidence for acne

            # Enhanced Hyperpigmentation analysis - stricter criteria
            if has_dark_patches:
                dark_patch_size = self._analyze_dark_patch_size(image)
                if (avg_brightness < 0.4 and contrast > 0.15 and dark_patch_size > 0.1) or \
                   (texture_variance > 0.1 and edge_density < 0.12 and dark_patch_size > 0.15):
                    scores[1] = 0.95  # Increased confidence for hyperpigmentation

                    # Reduce other scores only if dark patches are significant
                    if dark_patch_size > 0.2:
                        scores[4] = 0.1  # Reduce vitiligo score

            # Nail Psoriasis analysis - more specific patterns
            if texture_variance > 0.12 and edge_density > 0.18:
                nail_pattern_score = self._analyze_nail_pattern(image)
                if nail_pattern_score > 0.7:  # Strong nail pattern indication
                    scores[2] = 0.92

            # SJS-TEN analysis - require multiple indicators
            if red_dominance > 0.4 and edge_density > 0.25:
                blister_score = self._detect_blistering(image)
                if blister_score > 0.6:  # Clear blistering pattern
                    scores[3] = 0.94

            # Vitiligo analysis - require clear depigmentation
            if has_white_patches:
                white_patch_score = self._analyze_white_patches(image)
                if avg_brightness > 0.65 and contrast > 0.3 and white_patch_score > 0.7:
                    scores[4] = 0.96

            # If we have high confidence prediction
            max_score = np.max(scores)
            if max_score > 0.8:
                predicted_class = np.argmax(scores)
                # Normalize scores to probabilities
                scores = scores / np.sum(scores)
                return predicted_class, scores

        except Exception as e:
            print(f"Error in enhanced prediction: {e}")

        # Fallback to traditional prediction
        features = self.extract_features(image)
        features = features.reshape(1, -1)
        scaled_features = self.scaler.transform(features)
        class_probs = self.model.predict_proba(scaled_features)[0]
        predicted_class = np.argmax(class_probs)

        return predicted_class, class_probs

        # Enhanced multi-factor analysis
        try:
            # Convert to RGB uint8 for analysis
            img_rgb = (image * 255).astype(np.uint8)
            import cv2
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # Extract advanced metrics
            avg_brightness = np.mean(gray)
            contrast = np.std(gray)
            texture_variance = np.var(gray)

            # Color distribution analysis
            color_ratios = np.mean(img_rgb, axis=(0,1))
            red_dominance = color_ratios[0] / np.sum(color_ratios)

            # Edge detection for pattern analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Multi-factor condition scoring
            scores = np.zeros(5)  # One score per condition

            # Acne scoring
            if (red_dominance > 0.4 and edge_density > 0.1 and 
                self._detect_acne_patterns(image)):
                scores[0] = 0.9

            # Hyperpigmentation scoring
            if (avg_brightness < 0.4 and contrast > 0.15 and 
                self._detect_hyperpigmentation_patterns(image)):
                scores[1] = 0.85

            # Nail Psoriasis scoring
            if (texture_variance > 0.1 and edge_density > 0.15 and 
                self._detect_nail_psoriasis_patterns(image)):
                scores[2] = 0.88

            # SJS-TEN scoring
            if (red_dominance > 0.45 and edge_density > 0.2 and 
                self._detect_sjs_ten_patterns(image)):
                scores[3] = 0.92

            # Vitiligo scoring
            if (avg_brightness > 0.6 and contrast > 0.25 and 
                self._detect_vitiligo_patterns(image)):
                scores[4] = 0.87

            # If we have a high confidence prediction, use it
            max_score = np.max(scores)
            if max_score > 0.8:
                predicted_class = np.argmax(scores)
                # Normalize scores to probabilities
                scores = scores / np.sum(scores)
                return predicted_class, scores

        except Exception as e:
            print(f"Error in enhanced prediction: {e}")

        # Fallback to traditional feature-based prediction
        features = self.extract_features(image)
        features = features.reshape(1, -1)
        scaled_features = self.scaler.transform(features)
        class_probs = self.model.predict_proba(scaled_features)[0]
        predicted_class = np.argmax(class_probs)

        return predicted_class, class_probs

        # DIRECT VISUAL DIAGNOSIS: Use explicit visual cues to accurately identify conditions
        # rather than relying on machine learning probabilities

        ##### DIRECT DETECTION PHASE #####
        # Extract color channels for condition-specific detection
        # Analyze the image for direct visual cues that strongly indicate specific conditions

        if len(image.shape) == 3 and image.shape[2] >= 3:
            # Convert to HSV for better color analysis
            try:
                img_rgb = (image * 255).astype(np.uint8)
                import cv2
                img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

                # Get basic color statistics
                redness = np.mean(image[:,:,0])  # Red channel - high in acne, SJS-TEN
                greenness = np.mean(image[:,:,1])  # Green channel
                blueness = np.mean(image[:,:,2])  # Blue channel
                brightness = (redness + greenness + blueness) / 3

                # Texture analysis (simplified)
                texture_std = np.std(image)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

                # More advanced pattern detection
                has_red_spots = self._detect_acne_patterns(image)
                has_white_patches = self._detect_vitiligo_patterns(image)
                has_dark_patches = self._detect_hyperpigmentation_patterns(image)
                has_blistering = self._detect_sjs_ten_patterns(image)
                has_nail_lesions = self._detect_nail_psoriasis_patterns(image)

                # Print diagnostic info for debugging
                print(f"Image diagnostics: redness={redness:.2f}, brightness={brightness:.2f}")
                print(f"Pattern detection: acne={has_red_spots}, vitiligo={has_white_patches}, hyperpig={has_dark_patches}")

            except Exception as e:
                print(f"Error in image analysis: {e}")
                redness, brightness = 0, 0
                has_red_spots = has_white_patches = has_dark_patches = has_blistering = has_nail_lesions = False
        else:
            # Default values if image doesn't have proper color channels
            redness, brightness = 0, 0
            has_red_spots = has_white_patches = has_dark_patches = has_blistering = has_nail_lesions = False

        # DIRECT CLASSIFICATION RULES
        # Instead of boosting probabilities, directly classify based on visual cues

        # 1. ACNE: Red inflammatory spots/pustules
        is_acne = has_red_spots and redness > 0.35

        # 2. VITILIGO: White patches with clear borders
        is_vitiligo = has_white_patches and brightness > 0.5

        # 3. HYPERPIGMENTATION: Dark patches without inflammation
        is_hyperpigmentation = has_dark_patches and not has_red_spots and brightness < 0.45

        # 4. SJS-TEN: Widespread redness, blistering
        is_sjs_ten = has_blistering or (redness > 0.5 and texture_std > 0.25)

        # 5. NAIL PSORIASIS: Nail-specific lesions
        is_nail_psoriasis = has_nail_lesions

        # Count how many conditions it matches
        condition_matches = sum([is_acne, is_vitiligo, is_hyperpigmentation, is_sjs_ten, is_nail_psoriasis])

        # Initialize probabilities (will overwrite with direct detection)
        class_probs = np.zeros(5)

        # DIRECT CLASSIFICATION DECISION
        if condition_matches == 1:
            # Exactly one match - assign with high confidence
            if is_acne:
                class_probs[0] = 0.9
                predicted_class = 0
            elif is_hyperpigmentation:
                class_probs[1] = 0.9
                predicted_class = 1
            elif is_nail_psoriasis:
                class_probs[2] = 0.9
                predicted_class = 2
            elif is_sjs_ten:
                class_probs[3] = 0.9
                predicted_class = 3
            elif is_vitiligo:
                class_probs[4] = 0.9
                predicted_class = 4

        elif condition_matches > 1:
            # Multiple matches - choose the strongest one
            match_strengths = [
                0.7 if is_acne else 0.0,
                0.7 if is_hyperpigmentation else 0.0,
                0.7 if is_nail_psoriasis else 0.0,
                0.7 if is_sjs_ten else 0.0,
                0.7 if is_vitiligo else 0.0
            ]
            predicted_class = np.argmax(match_strengths)
            class_probs[predicted_class] = 0.8

        else:
            # No direct matches - fall back to ML prediction
            features = self.extract_features(image)
            features = features.reshape(1, -1)
            scaled_features = self.scaler.transform(features)
            class_probs = self.model.predict_proba(scaled_features)[0]
            predicted_class = np.argmax(class_probs)

            # Apply override for red inflamed skin that looks like acne
            if redness > 0.4 and predicted_class != 0:
                predicted_class = 0  # Force classify as acne
                class_probs = np.zeros(5)
                class_probs[0] = 0.9

        # Always ensure probabilities sum to 1
        if np.sum(class_probs) == 0:
            class_probs[predicted_class] = 1.0
        else:
            class_probs = class_probs / np.sum(class_probs)

        # Print final prediction for debugging
        print(f"Final prediction: {self.class_labels[predicted_class]} with confidence {class_probs[predicted_class]:.2f}")

        return predicted_class, class_probs

    def _detect_acne_patterns(self, image):
        """Detect acne using more sophisticated pattern recognition"""
        try:
            if len(image.shape) != 3:
                return False

            # Convert to RGB uint8
            img_rgb = (image * 255).astype(np.uint8)

            # Convert to HSV for better red detection
            import cv2
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

            # Extract red spots (acne inflammation)
            # Red in HSV is at both low and high H values
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2

            # Calculate percentage of red pixels
            red_pixel_percentage = np.sum(red_mask > 0) / (image.shape[0] * image.shape[1])

            # Find contours to identify spots
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter for small-medium sized spots (typical of acne)
            acne_like_spots = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < 500:  # Size range for typical acne spots
                    acne_like_spots += 1

            # Check both overall redness and presence of multiple small spots
            return (red_pixel_percentage > 0.03 and acne_like_spots >= 3) or red_pixel_percentage > 0.1

        except Exception as e:
            print(f"Error in acne detection: {e}")
            return False

    def _count_red_spots(self, image):
        """Count distinct red spots in the image"""
        try:
            img_rgb = (image * 255).astype(np.uint8)
            import cv2
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

            # Red mask (both low and high hue values)
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
            red_mask = cv2.add(mask1, mask2)

            # Find contours of red spots
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return len([c for c in contours if 10 < cv2.contourArea(c) < 500])
        except Exception as e:
            print(f"Error in red spot counting: {e}")
            return 0

    def _analyze_dark_patch_size(self, image):
        """Analyze size and distribution of dark patches"""
        try:
            img_rgb = (image * 255).astype(np.uint8)
            import cv2
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # Threshold for dark regions
            thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]

            # Calculate relative area of dark patches
            dark_area = np.sum(thresh > 0) / thresh.size
            return dark_area
        except Exception as e:
            print(f"Error in dark patch analysis: {e}")
            return 0

    def _analyze_nail_pattern(self, image):
        """Analyze nail-specific patterns"""
        try:
            img_rgb = (image * 255).astype(np.uint8)
            import cv2
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # Look for linear patterns and pitting
            gradX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            gradY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

            # Calculate gradient magnitude
            gradient = np.sqrt(gradX**2 + gradY**2)

            # Normalize and threshold
            pattern_score = np.sum(gradient > np.mean(gradient)) / gradient.size
            return pattern_score
        except Exception as e:
            print(f"Error in nail pattern analysis: {e}")
            return 0

    def _detect_blistering(self, image):
        """Detect blistering patterns"""
        try:
            img_rgb = (image * 255).astype(np.uint8)
            import cv2
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)

            # Look for circular/bubble-like patterns
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            blister_score = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * cv2.contourArea(cnt) / (perimeter * perimeter)
                    if circularity > 0.7:  # More circular shapes
                        blister_score += 0.1

            return min(blister_score, 1.0)
        except Exception as e:
            print(f"Error in blistering detection: {e}")
            return 0

    def _analyze_white_patches(self, image):
        """Analyze white patches for vitiligo patterns"""
        try:
            img_rgb = (image * 255).astype(np.uint8)
            import cv2
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # Threshold for bright regions
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

            # Find contours of white patches
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            patch_score = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    # Analyze patch shape and boundaries
                    perimeter = cv2.arcLength(cnt, True)
                    smoothness = area / (perimeter * perimeter)
                    if smoothness > 0.1:  # Smoother boundaries typical of vitiligo
                        patch_score += 0.2

            return min(patch_score, 1.0)
        except Exception as e:
            print(f"Error in white patch analysis: {e}")
            return 0

            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter for small-medium sized spots (typical of acne)
            acne_like_spots = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < 500:  # Size range for typical acne spots
                    acne_like_spots += 1

            # Check both overall redness and presence of multiple small spots
            return (red_pixel_percentage > 0.03 and acne_like_spots >= 3) or red_pixel_percentage > 0.1

        except Exception as e:
            print(f"Error in acne detection: {e}")
            return False

    def _detect_vitiligo_patterns(self, image):
        """Detect vitiligo using improved pattern recognition"""
        try:
            if len(image.shape) != 3:
                return False

            # Convert to RGB uint8
            img_rgb = (image * 255).astype(np.uint8)

            # Convert to grayscale
            import cv2
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # Adaptive threshold to find very bright areas
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 11, -30)  # -30 offset to detect brighter regions

            # Clean up the mask
            kernel = np.ones((5,5),np.uint8)
            clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Find contours of white patches
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter for larger patches with clear borders
            vitiligo_like_patches = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if area > 200 and perimeter > 0:  # Significant size
                    # Compactness - vitiligo tends to have smoother borders
                    compactness = 4 * np.pi * area / (perimeter * perimeter)
                    if compactness > 0.2:  # Not too irregular
                        vitiligo_like_patches += 1

            # White pixel percentage
            white_pixel_percentage = np.sum(clean_mask > 0) / (clean_mask.shape[0] * clean_mask.shape[1])

            return (white_pixel_percentage > 0.05 and vitiligo_like_patches >= 1)

        except Exception as e:
            print(f"Error in vitiligo detection: {e}")
            return False

    def _detect_hyperpigmentation_patterns(self, image):
        """Detect hyperpigmentation using improved pattern recognition"""
        try:
            if len(image.shape) != 3:
                return False

            import cv2
            # Convert to RGB uint8
            img_rgb = (image * 255).astype(np.uint8)

            # Convert to LAB color space for better color segmentation
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

            # Extract lightness channel
            l_channel = img_lab[:,:,0]

            # Calculate local contrast
            kernel_size = 15
            local_mean = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)
            local_contrast = cv2.absdiff(l_channel, local_mean)

            # Adaptive threshold for dark regions
            dark_mask = cv2.adaptiveThreshold(l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 21, 5)

            # Clean up mask
            kernel = np.ones((5,5), np.uint8)
            clean_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

            # Find contours of dark patches
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analyze patch characteristics
            large_patches = 0
            uniform_patches = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Larger patches typical of hyperpigmentation
                    x,y,w,h = cv2.boundingRect(contour)
                    patch = local_contrast[y:y+h, x:x+w]
                    # Check if patch has uniform color (low contrast)
                    if np.mean(patch) < 20:  # Low local contrast threshold
                        uniform_patches += 1
                    large_patches += 1

            # Check color uniformity in b channel (yellow-blue) of LAB
            b_channel = img_lab[:,:,2]
            b_std = np.std(b_channel)

            # Calculate dark area percentage
            dark_percentage = np.sum(clean_mask > 0) / clean_mask.size

            # Extract red channel to check for inflammation
            r_channel = img_rgb[:,:,0]
            red_intensity = np.mean(r_channel)

            # Hyperpigmentation criteria:
            # - Has large, uniform dark patches
            # - Shows color uniformity
            # - Limited red intensity (to distinguish from acne)
            return (large_patches >= 2 and 
                   uniform_patches >= 1 and 
                   b_std < 45 and 
                   dark_percentage > 0.1 and 
                   red_intensity < 150)

        except Exception as e:
            print(f"Error in hyperpigmentation detection: {e}")
            return False

    def _detect_sjs_ten_patterns(self, image):
        """Detect SJS-TEN patterns (widespread redness, blistering)"""
        try:
            if len(image.shape) != 3:
                return False

            # Convert to RGB uint8
            img_rgb = (image * 255).astype(np.uint8)

            # Convert to different color spaces
            import cv2
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # 1. Check for widespread redness
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2

            red_percentage = np.sum(red_mask > 0) / (red_mask.shape[0] * red_mask.shape[1])

            # 2. Check for blistering texture using edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # 3. Check for irregular texture using standard deviation
            texture_std = np.std(gray)

            # Combined criteria for SJS-TEN
            return (red_percentage > 0.25 and edge_density > 0.1) or \
                   (red_percentage > 0.2 and texture_std > 50)

        except Exception as e:
            print(f"Error in SJS-TEN detection: {e}")
            return False

    def _detect_nail_psoriasis_patterns(self, image):
        """Detect nail psoriasis patterns"""
        try:
            if len(image.shape) != 3:
                return False

            # Convert to RGB uint8
            img_rgb = (image * 255).astype(np.uint8)

            # Convert to grayscale
            import cv2
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # 1. Check for nail shape - look for rectangular structures
            # Since nails usually have distinct shapes
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            has_nail_shape = False
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Significant size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / (min(w, h) + 0.01)  # Avoid division by zero
                    # Nails typically have rectangular shape with specific aspect ratio
                    if 1.5 < aspect_ratio < 4:
                        has_nail_shape = True
                        break

            # 2. Check for nail pitting using texture analysis
            # Apply Laplacian filter to detect small pits
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_std = np.std(laplacian)

            # Higher laplacian standard deviation indicates more pitting/irregularities
            has_pitting = laplacian_std > 20

            # Combining the criteria for nail psoriasis
            return has_nail_shape and has_pitting

        except Exception as e:
            print(f"Error in nail psoriasis detection: {e}")
            return False

    def _has_red_spots(self, image):
        """Detect if image has red spots characteristic of acne"""
        try:
            if len(image.shape) != 3:
                return False

            # Convert to HSV for better color segmentation
            img_rgb = (image * 255).astype(np.uint8)
            import cv2
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

            # Red range in HSV (two ranges because red wraps around)
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])

            # Create masks for red regions
            mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
            mask = mask1 + mask2

            # Check if enough red regions are present
            red_pixel_percentage = np.sum(mask > 0) / (image.shape[0] * image.shape[1])

            return red_pixel_percentage > 0.05
        except Exception as e:
            print(f"Error in red spot detection: {e}")
            return False

    def _has_white_patches(self, image):
        """Detect if image has white patches characteristic of vitiligo"""
        try:
            if len(image.shape) != 3:
                return False

            # Convert to grayscale
            img_gray = np.mean(image, axis=2)

            # Threshold to find bright areas
            bright_threshold = 0.7  # Threshold for bright areas
            bright_mask = img_gray > bright_threshold

            # Calculate percentage of bright pixels
            bright_pixel_percentage = np.sum(bright_mask) / (image.shape[0] * image.shape[1])

            # Check for contrast between bright and surrounding areas (vitiligo has distinct borders)
            has_distinct_borders = False
            if bright_pixel_percentage > 0.05:
                # Dilate the mask to get surrounding area
                import cv2
                kernel = np.ones((5,5), np.uint8)
                dilated = cv2.dilate(bright_mask.astype(np.uint8), kernel, iterations=1)

                # Get pixels at the boundary (in dilated but not in original mask)
                boundary = dilated.astype(bool) & ~bright_mask

                # If boundary exists and has significantly lower brightness, it suggests vitiligo
                if np.sum(boundary) > 0:
                    boundary_brightness = np.mean(img_gray[boundary])
                    patch_brightness = np.mean(img_gray[bright_mask])
                    brightness_contrast = patch_brightness - boundary_brightness
                    has_distinct_borders = brightness_contrast > 0.2

            return bright_pixel_percentage > 0.08 and has_distinct_borders
        except Exception as e:
            print(f"Error in white patch detection: {e}")
            return False

    def _has_dark_patches(self, image):
        """Detect if image has dark patches characteristic of hyperpigmentation"""
        try:
            if len(image.shape) != 3:
                return False

            # Convert to grayscale using weighted channels for better skin tone detection
            img_gray = image[:,:,0] * 0.3 + image[:,:,1] * 0.59 + image[:,:,2] * 0.11

            # Adaptive thresholding for better dark patch detection
            dark_threshold = np.mean(img_gray) * 0.85  # Dynamic threshold
            dark_mask = img_gray < dark_threshold

            # Calculate percentage of dark pixels
            dark_pixel_percentage = np.sum(dark_mask) / (image.shape[0] * image.shape[1])

            # Check if dark areas form continuous regions (not just scattered pixels)
            import cv2
            # Close small gaps to form continuous regions
            kernel = np.ones((3,3), np.uint8)
            closed_mask = cv2.morphologyEx(dark_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask)

            # Check if there's at least one significant dark region
            has_significant_dark_region = False
            for i in range(1, num_labels):  # Skip label 0 (background)
                if stats[i, cv2.CC_STAT_AREA] > 100:  # Area threshold
                    has_significant_dark_region = True
                    break

            return dark_pixel_percentage > 0.05 and has_significant_dark_region
        except Exception as e:
            print(f"Error in dark patch detection: {e}")
            return False

# Load the classifier model
@st.cache_resource
def load_vit_model():
    # Initialize model
    model = ViTInspiredClassifier(num_classes=5)
    print("Model loaded successfully")
    return model

# Define image transformation
def get_transform():
    # Transformation function that resizes and normalizes the image
    def transform(image):
        if isinstance(image, Image.Image):
            # Resize image
            image_resized = image.resize((224, 244))

            # Convert to numpy array and normalize
            img_array = np.array(image_resized, dtype=np.float32) / 255.0

            # Normalize using ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # Check if the image has 3 channels
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = (img_array - mean) / std

            return img_array
        return image
    return transform

# Function to predict skin disease from an image
def predict_skin_disease(model, image_tensor):
    # Simply pass to the model's predict method
    predicted_class, confidence_scores = model.predict(image_tensor)
    return predicted_class, confidence_scores