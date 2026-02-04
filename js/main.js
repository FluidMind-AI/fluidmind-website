/**
 * FLUID MIND — WebGL Metaball Lava Lamp
 *
 * True scalar field metaballs rendered per-pixel in fragment shader.
 * Smooth isosurface with volumetric shading + multi-pass bloom.
 */

(function() {
    'use strict';

    // Reduced motion check
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const canvas = document.getElementById('lava');
    if (!canvas) return;

    // =============================================
    // CONFIGURATION
    // =============================================
    const CONFIG = {
        // Blobs
        numBlobs: 12,
        minRadius: 0.06,  // as fraction of screen height
        maxRadius: 0.22,

        // Physics - slow for testing
        speed: 0.000018,
        buoyancy: 0.000004,
        wanderStrength: 0.00001,

        // Collision physics
        collisionRange: 1.18,  // Multiplier on (r1 + r2) to start interaction
        collisionStrength: 0.00012,
        collisionStickiness: 0.00003, // Mild attraction to keep blobs together
        collisionDamping: 0.95,

        // Metaball rendering
        threshold: 0.5,
        edgeSoftness: 0.18,    // Smoothstep range

        // Color (bright cyan)
        hue: 0.52,             // 0-1 (shifted towards cyan)

        // Bloom (brighter glow)
        bloomThreshold: 0.25,
        bloomIntensity: 0.5,
        bloomRadius: 0.018,    // As fraction of screen
    };

    // =============================================
    // WEBGL SETUP
    // =============================================
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) {
        console.warn('WebGL not supported, falling back to static');
        return;
    }

    let width, height, dpr;
    let blobs = [];
    let program, bloomProgram, compositeProgram;
    let framebuffers = {};
    let textures = {};
    let quadVAO;

    // Pre-allocated array to avoid garbage collection hitches
    const blobData = new Float32Array(72); // 24 blobs * 3 floats

    // =============================================
    // SHADERS
    // =============================================

    // Vertex shader (fullscreen quad)
    const vertexShaderSource = `#version 300 es
        in vec2 a_position;
        out vec2 v_uv;
        void main() {
            v_uv = a_position * 0.5 + 0.5;
            gl_Position = vec4(a_position, 0.0, 1.0);
        }
    `;

    // Metaball fragment shader
    const metaballFragmentSource = `#version 300 es
        precision highp float;

        in vec2 v_uv;
        out vec4 fragColor;

        uniform vec2 u_resolution;
        uniform float u_time;
        uniform vec3 u_blobs[24]; // x, y, radius (max 24 blobs)
        uniform int u_numBlobs;
        uniform float u_threshold;
        uniform float u_softness;
        uniform float u_hue;

        // HSL to RGB
        vec3 hsl2rgb(float h, float s, float l) {
            vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
            return l + s * (rgb - 0.5) * (1.0 - abs(2.0 * l - 1.0));
        }

        void main() {
            vec2 uv = v_uv;
            float aspect = u_resolution.x / u_resolution.y;
            vec2 pos = vec2(uv.x * aspect, uv.y);

            // Calculate metaball field
            float field = 0.0;
            for (int i = 0; i < 24; i++) {
                if (i >= u_numBlobs) break;
                vec2 blobPos = vec2(u_blobs[i].x * aspect, u_blobs[i].y);
                float radius = u_blobs[i].z;
                float dist = distance(pos, blobPos);

                // Smooth falloff: r² / (d² + small) for continuous field
                field += (radius * radius) / (dist * dist + 0.0001);
            }

            // Isosurface with smoothstep (no hard edges)
            float surface = smoothstep(u_threshold - u_softness, u_threshold + u_softness, field);

            if (surface < 0.01) {
                // Outside - bright aqua-tinted background
                fragColor = vec4(0.94, 0.984, 1.0, 1.0);
                return;
            }

            // Volumetric shading based on field strength
            // Normalize field for shading (how "deep" inside the blob)
            float depth = smoothstep(u_threshold, u_threshold + 2.0, field);

            // Lightness: brighter overall, still darker center
            float lightness = 0.50 + (1.0 - depth) * 0.32;
            lightness = clamp(lightness, 0.45, 0.85);

            // Bright rim glow
            float rim = 1.0 - smoothstep(0.0, 0.5, surface - 0.1);
            lightness = clamp(lightness + rim * 0.12, 0.45, 0.90);

            // Higher saturation for vibrant colors
            float saturation = 0.65 + depth * 0.2;

            vec3 color = hsl2rgb(u_hue, saturation, lightness);

            // Apply surface alpha for anti-aliased edges
            fragColor = vec4(color, surface * 0.75);
        }
    `;

    // Bloom extract + blur shader
    const bloomFragmentSource = `#version 300 es
        precision highp float;

        in vec2 v_uv;
        out vec4 fragColor;

        uniform sampler2D u_texture;
        uniform vec2 u_direction; // (1,0) for horizontal, (0,1) for vertical
        uniform float u_radius;
        uniform vec2 u_resolution;
        uniform float u_threshold;
        uniform bool u_extract;

        void main() {
            vec2 texelSize = 1.0 / u_resolution;
            vec4 color = vec4(0.0);

            if (u_extract) {
                // Extract bright areas
                vec4 c = texture(u_texture, v_uv);
                float brightness = dot(c.rgb, vec3(0.299, 0.587, 0.114));
                if (brightness > u_threshold) {
                    fragColor = c * (brightness - u_threshold) / (1.0 - u_threshold);
                } else {
                    fragColor = vec4(0.0);
                }
                return;
            }

            // 9-tap Gaussian blur
            float weights[5];
            weights[0] = 0.227027;
            weights[1] = 0.1945946;
            weights[2] = 0.1216216;
            weights[3] = 0.054054;
            weights[4] = 0.016216;

            vec2 dir = u_direction * texelSize * u_radius;

            color += texture(u_texture, v_uv) * weights[0];
            for (int i = 1; i < 5; i++) {
                color += texture(u_texture, v_uv + dir * float(i)) * weights[i];
                color += texture(u_texture, v_uv - dir * float(i)) * weights[i];
            }

            fragColor = color;
        }
    `;

    // Final composite shader
    const compositeFragmentSource = `#version 300 es
        precision highp float;

        in vec2 v_uv;
        out vec4 fragColor;

        uniform sampler2D u_base;
        uniform sampler2D u_bloom;
        uniform float u_bloomIntensity;

        void main() {
            vec4 base = texture(u_base, v_uv);
            vec4 bloom = texture(u_bloom, v_uv);

            // Additive bloom
            vec3 color = base.rgb + bloom.rgb * u_bloomIntensity;

            // Slight tone mapping to prevent blowout
            color = color / (color + vec3(0.5));

            fragColor = vec4(color, 1.0);
        }
    `;

    // =============================================
    // BLOB CLASS
    // =============================================
    class Blob {
        constructor(x, y, radius, index) {
            this.x = x;
            this.y = y;
            this.radius = radius;
            this.vx = (Math.random() - 0.5) * CONFIG.speed * 2;
            this.vy = (Math.random() - 0.5) * CONFIG.speed;
            this.ax = 0;
            this.ay = 0;
            this.wanderAngle = Math.random() * Math.PI * 2;
            this.mass = radius / CONFIG.maxRadius;
            // Unique phase offsets for smooth noise-based wandering
            this.phaseX = Math.random() * 1000;
            this.phaseY = Math.random() * 1000;
            this.wanderSpeed = 0.000015 + Math.random() * 0.00001;
        }

        update(dt, time) {
            // Buoyancy (smaller rises faster)
            this.vy -= CONFIG.buoyancy * (1.0 - this.mass * 0.6);

            // Smooth wandering using sine waves instead of random
            const wanderX = Math.sin(time * this.wanderSpeed + this.phaseX);
            const wanderY = Math.sin(time * this.wanderSpeed * 0.7 + this.phaseY);
            this.vx += wanderX * CONFIG.wanderStrength;
            this.vy += wanderY * CONFIG.wanderStrength * 0.3;

            // Apply collision forces accumulated this frame
            this.vx += this.ax;
            this.vy += this.ay;
            this.ax = 0;
            this.ay = 0;

            // Damping (viscosity)
            this.vx *= 0.995 * CONFIG.collisionDamping;
            this.vy *= 0.995 * CONFIG.collisionDamping;

            // Clamp speed
            const speed = Math.hypot(this.vx, this.vy);
            const maxSpeed = CONFIG.speed * 3;
            if (speed > maxSpeed) {
                this.vx = (this.vx / speed) * maxSpeed;
                this.vy = (this.vy / speed) * maxSpeed;
            }

            // Move
            this.x += this.vx * dt;
            this.y += this.vy * dt;

            // Wrap
            const pad = this.radius * 2;
            if (this.y < -pad) { this.y = 1 + pad; this.x = Math.random(); }
            if (this.y > 1 + pad) { this.y = -pad; this.x = Math.random(); }
            if (this.x < -pad) this.x = 1 + pad;
            if (this.x > 1 + pad) this.x = -pad;
        }
    }

    // =============================================
    // GL HELPERS
    // =============================================
    function createShader(type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    function createProgram(vsSource, fsSource) {
        const vs = createShader(gl.VERTEX_SHADER, vsSource);
        const fs = createShader(gl.FRAGMENT_SHADER, fsSource);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(prog));
            return null;
        }
        return prog;
    }

    function createFramebuffer(w, h) {
        const fb = gl.createFramebuffer();
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return { framebuffer: fb, texture: tex };
    }

    // =============================================
    // INIT
    // =============================================
    function init() {
        resize();
        createBlobs();
        setupGL();
        window.addEventListener('resize', debounce(resize, 200));

        // Scroll animations (GSAP ScrollTrigger)
        initScrollAnimations();

        // Carousel arrow navigation
        initCarouselArrows();

        // Nav scroll effect
        initNavScroll();

        // Mobile menu
        initMobileMenu();

        // Smooth scroll with centering
        initSmoothScroll();

        // Dynamic year
        initYear();

        if (prefersReducedMotion) {
            // Render one frame
            render(0);
        } else {
            requestAnimationFrame(animate);
        }
    }

    function resize() {
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        width = window.innerWidth;
        height = window.innerHeight;

        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';

        gl.viewport(0, 0, canvas.width, canvas.height);

        // Recreate framebuffers at new size
        const fbWidth = Math.floor(canvas.width * 0.5);  // Half res for bloom
        const fbHeight = Math.floor(canvas.height * 0.5);

        framebuffers.base = createFramebuffer(canvas.width, canvas.height);
        framebuffers.bloomA = createFramebuffer(fbWidth, fbHeight);
        framebuffers.bloomB = createFramebuffer(fbWidth, fbHeight);
    }

    function createBlobs() {
        blobs = [];

        // Large (2-3)
        for (let i = 0; i < 3; i++) {
            blobs.push(new Blob(
                Math.random(),
                Math.random(),
                CONFIG.maxRadius * (0.7 + Math.random() * 0.3)
            ));
        }

        // Medium (5)
        for (let i = 0; i < 5; i++) {
            blobs.push(new Blob(
                Math.random(),
                Math.random(),
                CONFIG.minRadius + (CONFIG.maxRadius - CONFIG.minRadius) * (0.35 + Math.random() * 0.35)
            ));
        }

        // Small (4)
        for (let i = 0; i < 4; i++) {
            blobs.push(new Blob(
                Math.random(),
                Math.random(),
                CONFIG.minRadius + (CONFIG.maxRadius - CONFIG.minRadius) * Math.random() * 0.35
            ));
        }
    }

    function setupGL() {
        // Create programs
        program = createProgram(vertexShaderSource, metaballFragmentSource);
        bloomProgram = createProgram(vertexShaderSource, bloomFragmentSource);
        compositeProgram = createProgram(vertexShaderSource, compositeFragmentSource);

        // Fullscreen quad
        const quadVerts = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
        const quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);

        // Store for drawing
        quadVAO = quadBuffer;

        // Enable blending
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }

    function drawQuad(prog) {
        const posLoc = gl.getAttribLocation(prog, 'a_position');
        gl.bindBuffer(gl.ARRAY_BUFFER, quadVAO);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    // =============================================
    // SCROLL ANIMATIONS (GSAP ScrollTrigger)
    // =============================================
    function initScrollAnimations() {
        // Skip animations for reduced motion preference
        if (prefersReducedMotion) {
            // Just show everything immediately
            document.querySelectorAll('.flip-card, .reveal').forEach(el => {
                el.classList.add('visible');
            });
            return;
        }

        // Check if GSAP is loaded
        if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') {
            // Fallback to simple IntersectionObserver
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                    }
                });
            }, { threshold: 0.15, rootMargin: '0px 0px -50px 0px' });

            document.querySelectorAll('.flip-card, .reveal').forEach(el => {
                observer.observe(el);
            });
            return;
        }

        // Register ScrollTrigger plugin
        gsap.registerPlugin(ScrollTrigger);

        // Hero content - fade up on load
        gsap.from('.hero h1', {
            y: 60,
            opacity: 0,
            duration: 1.2,
            ease: 'power3.out',
            delay: 0.3
        });

        gsap.from('.hero-sub', {
            y: 40,
            opacity: 0,
            duration: 1,
            ease: 'power3.out',
            delay: 0.6
        });

        // Our Expertise heading
        gsap.from('.capabilities-heading', {
            y: 40,
            opacity: 0,
            duration: 1,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: '.capabilities',
                start: 'top 75%',
                toggleActions: 'play none none reverse'
            }
        });

        // Flip cards - soft fade in
        gsap.from('.flip-card', {
            y: 20,
            opacity: 0,
            duration: 1,
            stagger: 0.1,
            ease: 'power2.out',
            scrollTrigger: {
                trigger: '.carousel',
                start: 'top 85%',
                toggleActions: 'play none none reverse'
            }
        });

        // Who We Are section
        gsap.from('.who-left', {
            x: -40,
            opacity: 0,
            duration: 1,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: '.who-we-are',
                start: 'top 70%',
                toggleActions: 'play none none reverse'
            }
        });

        gsap.from('.who-right', {
            x: 40,
            opacity: 0,
            duration: 1,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: '.who-we-are',
                start: 'top 70%',
                toggleActions: 'play none none reverse'
            },
            delay: 0.2
        });

        gsap.from('.who-traits span', {
            y: 20,
            opacity: 0,
            duration: 0.6,
            stagger: 0.1,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: '.who-traits',
                start: 'top 85%',
                toggleActions: 'play none none reverse'
            }
        });

        // Manifesto - dramatic staggered reveal
        gsap.utils.toArray('.manifesto-content p').forEach((p, i) => {
            gsap.from(p, {
                y: 50,
                opacity: 0,
                duration: 1.2,
                ease: 'power2.out',
                scrollTrigger: {
                    trigger: '.manifesto',
                    start: 'top 60%',
                    toggleActions: 'play none none reverse'
                },
                delay: i * 0.25
            });
        });

        // Manifesto highlight - special treatment
        gsap.from('.manifesto .highlight', {
            scale: 0.9,
            scrollTrigger: {
                trigger: '.manifesto',
                start: 'top 50%',
                toggleActions: 'play none none reverse'
            },
            duration: 0.8,
            ease: 'back.out(1.7)',
            delay: 0.7
        });

        // Contact section
        gsap.from('.contact h2', {
            y: 40,
            opacity: 0,
            duration: 1,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: '.contact',
                start: 'top 70%',
                toggleActions: 'play none none reverse'
            }
        });

        gsap.from('.contact-link', {
            y: 30,
            opacity: 0,
            duration: 0.8,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: '.contact',
                start: 'top 70%',
                toggleActions: 'play none none reverse'
            },
            delay: 0.3
        });
    }

    // =============================================
    // NAV SCROLL EFFECT
    // =============================================
    function initNavScroll() {
        const nav = document.querySelector('.nav');
        if (!nav) return;

        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                nav.classList.add('scrolled');
            } else {
                nav.classList.remove('scrolled');
            }
        });
    }

    // =============================================
    // MOBILE MENU
    // =============================================
    function initMobileMenu() {
        const burger = document.querySelector('.nav-burger');
        const navLinks = document.querySelector('.nav-links');
        if (!burger || !navLinks) return;

        burger.addEventListener('click', () => {
            burger.classList.toggle('active');
            navLinks.classList.toggle('open');
            burger.setAttribute('aria-expanded', navLinks.classList.contains('open'));
        });

        // Close menu when clicking a link
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                burger.classList.remove('active');
                navLinks.classList.remove('open');
                burger.setAttribute('aria-expanded', 'false');
            });
        });
    }

    // =============================================
    // DYNAMIC YEAR
    // =============================================
    function initYear() {
        const yearEl = document.getElementById('year');
        if (yearEl) {
            yearEl.textContent = new Date().getFullYear();
        }
    }

    // =============================================
    // SMOOTH SCROLL WITH CENTERING
    // =============================================
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const href = this.getAttribute('href');
                if (href === '#') return;

                const target = document.querySelector(href);
                if (!target) return;

                e.preventDefault();

                const targetRect = target.getBoundingClientRect();
                const targetHeight = target.offsetHeight;
                const windowHeight = window.innerHeight;

                // Calculate offset to center the section (or near-center for tall sections)
                let offset;
                if (targetHeight < windowHeight * 0.8) {
                    // Section fits on screen - center it
                    offset = (windowHeight - targetHeight) / 2 - 60;
                } else {
                    // Section is tall - just add some top padding
                    offset = 100;
                }

                const scrollTo = window.scrollY + targetRect.top - offset;

                window.scrollTo({
                    top: Math.max(0, scrollTo),
                    behavior: 'smooth'
                });
            });
        });
    }

    // =============================================
    // CAROUSEL ARROWS + AUTO-SCROLL
    // =============================================
    function initCarouselArrows() {
        const carousel = document.querySelector('.carousel');
        const leftArrow = document.querySelector('.carousel-arrow-left');
        const rightArrow = document.querySelector('.carousel-arrow-right');

        if (!carousel || !leftArrow || !rightArrow) return;

        const scrollAmount = 320; // Card width + gap

        leftArrow.addEventListener('click', () => {
            carousel.scrollBy({ left: -scrollAmount, behavior: 'smooth' });
        });

        rightArrow.addEventListener('click', () => {
            carousel.scrollBy({ left: scrollAmount, behavior: 'smooth' });
        });

        // Auto-scroll
        let autoScrollSpeed = 0.5; // pixels per frame
        let isHovered = false;
        let animationId;

        function autoScroll() {
            if (!isHovered) {
                carousel.scrollLeft += autoScrollSpeed;

                // Loop back when reaching the end
                if (carousel.scrollLeft >= carousel.scrollWidth - carousel.clientWidth) {
                    carousel.scrollLeft = 0;
                }
            }
            animationId = requestAnimationFrame(autoScroll);
        }

        // Start auto-scroll
        autoScroll();

        // Pause on hover
        carousel.addEventListener('mouseenter', () => {
            isHovered = true;
        });

        carousel.addEventListener('mouseleave', () => {
            isHovered = false;
        });

        // Also pause when clicking arrows
        leftArrow.addEventListener('mouseenter', () => { isHovered = true; });
        leftArrow.addEventListener('mouseleave', () => { isHovered = false; });
        rightArrow.addEventListener('mouseenter', () => { isHovered = true; });
        rightArrow.addEventListener('mouseleave', () => { isHovered = false; });
    }

    // =============================================
    // RENDER
    // =============================================
    let lastTime = 0;
    const FIXED_DT = 16.667; // Fixed 60fps timestep

    function animate(time) {
        // Collision impulses (simple springy separation)
        for (let i = 0; i < blobs.length; i++) {
            for (let j = i + 1; j < blobs.length; j++) {
                const a = blobs[i];
                const b = blobs[j];
                const dx = b.x - a.x;
                const dy = b.y - a.y;
                const dist = Math.hypot(dx, dy) + 0.00001;
                const minDist = (a.radius + b.radius) * CONFIG.collisionRange;

                if (dist < minDist) {
                    const overlap = (minDist - dist) / minDist;
                    const nx = dx / dist;
                    const ny = dy / dist;
                    const force = overlap * overlap * CONFIG.collisionStrength;

                    a.ax -= nx * force * (1.0 - a.mass * 0.4);
                    a.ay -= ny * force * (1.0 - a.mass * 0.4);
                    b.ax += nx * force * (1.0 - b.mass * 0.4);
                    b.ay += ny * force * (1.0 - b.mass * 0.4);

                    // Mild attraction to keep blobs connected longer
                    const stick = CONFIG.collisionStickiness * overlap;
                    a.ax += nx * stick;
                    a.ay += ny * stick;
                    b.ax -= nx * stick;
                    b.ay -= ny * stick;

                }
            }
        }

        // Use fixed timestep for consistent movement
        blobs.forEach(b => b.update(FIXED_DT, time));

        render(time);
        requestAnimationFrame(animate);
    }

    function render(time) {
        // 1. Render metaballs to base framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.base.framebuffer);
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.94, 0.984, 1.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);

        // Set uniforms
        gl.uniform2f(gl.getUniformLocation(program, 'u_resolution'), canvas.width, canvas.height);
        gl.uniform1f(gl.getUniformLocation(program, 'u_time'), time * 0.001);
        gl.uniform1f(gl.getUniformLocation(program, 'u_threshold'), CONFIG.threshold);
        gl.uniform1f(gl.getUniformLocation(program, 'u_softness'), CONFIG.edgeSoftness);
        gl.uniform1f(gl.getUniformLocation(program, 'u_hue'), CONFIG.hue);
        gl.uniform1i(gl.getUniformLocation(program, 'u_numBlobs'), blobs.length);

        // Pack blob data into pre-allocated array
        for (let i = 0; i < blobs.length; i++) {
            blobData[i * 3] = blobs[i].x;
            blobData[i * 3 + 1] = blobs[i].y;
            blobData[i * 3 + 2] = blobs[i].radius;
        }
        gl.uniform3fv(gl.getUniformLocation(program, 'u_blobs'), blobData);

        drawQuad(program);

        // 2. Extract bright areas for bloom
        const bloomW = Math.floor(canvas.width * 0.5);
        const bloomH = Math.floor(canvas.height * 0.5);

        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.bloomA.framebuffer);
        gl.viewport(0, 0, bloomW, bloomH);
        gl.useProgram(bloomProgram);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, framebuffers.base.texture);
        gl.uniform1i(gl.getUniformLocation(bloomProgram, 'u_texture'), 0);
        gl.uniform2f(gl.getUniformLocation(bloomProgram, 'u_resolution'), bloomW, bloomH);
        gl.uniform1f(gl.getUniformLocation(bloomProgram, 'u_threshold'), CONFIG.bloomThreshold);
        gl.uniform1i(gl.getUniformLocation(bloomProgram, 'u_extract'), 1);
        gl.uniform2f(gl.getUniformLocation(bloomProgram, 'u_direction'), 0, 0);
        gl.uniform1f(gl.getUniformLocation(bloomProgram, 'u_radius'), 1);

        drawQuad(bloomProgram);

        // 3. Horizontal blur
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.bloomB.framebuffer);
        gl.bindTexture(gl.TEXTURE_2D, framebuffers.bloomA.texture);
        gl.uniform1i(gl.getUniformLocation(bloomProgram, 'u_extract'), 0);
        gl.uniform2f(gl.getUniformLocation(bloomProgram, 'u_direction'), 1, 0);
        gl.uniform1f(gl.getUniformLocation(bloomProgram, 'u_radius'), CONFIG.bloomRadius * canvas.width);

        drawQuad(bloomProgram);

        // 4. Vertical blur
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.bloomA.framebuffer);
        gl.bindTexture(gl.TEXTURE_2D, framebuffers.bloomB.texture);
        gl.uniform2f(gl.getUniformLocation(bloomProgram, 'u_direction'), 0, 1);

        drawQuad(bloomProgram);

        // 5. Second blur pass for wider glow
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.bloomB.framebuffer);
        gl.bindTexture(gl.TEXTURE_2D, framebuffers.bloomA.texture);
        gl.uniform2f(gl.getUniformLocation(bloomProgram, 'u_direction'), 1, 0);
        gl.uniform1f(gl.getUniformLocation(bloomProgram, 'u_radius'), CONFIG.bloomRadius * canvas.width * 2);

        drawQuad(bloomProgram);

        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.bloomA.framebuffer);
        gl.bindTexture(gl.TEXTURE_2D, framebuffers.bloomB.texture);
        gl.uniform2f(gl.getUniformLocation(bloomProgram, 'u_direction'), 0, 1);

        drawQuad(bloomProgram);

        // 6. Composite to screen
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(compositeProgram);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, framebuffers.base.texture);
        gl.uniform1i(gl.getUniformLocation(compositeProgram, 'u_base'), 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, framebuffers.bloomA.texture);
        gl.uniform1i(gl.getUniformLocation(compositeProgram, 'u_bloom'), 1);

        gl.uniform1f(gl.getUniformLocation(compositeProgram, 'u_bloomIntensity'), CONFIG.bloomIntensity);

        drawQuad(compositeProgram);
    }

    // =============================================
    // UTILS
    // =============================================
    function debounce(fn, wait) {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => fn(...args), wait);
        };
    }

    // =============================================
    // START
    // =============================================
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Pause when hidden
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && !prefersReducedMotion) {
            lastTime = performance.now();
            requestAnimationFrame(animate);
        }
    });

})();
