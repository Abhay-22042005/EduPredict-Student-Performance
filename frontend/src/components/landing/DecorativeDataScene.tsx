import { Canvas, useFrame } from "@react-three/fiber";
import { useReducedMotion } from "framer-motion";
import { Float, OrbitControls, Sparkles } from "@react-three/drei";
import {
  Component,
  Suspense,
  useMemo,
  useRef,
  useSyncExternalStore,
} from "react";
import type { ReactNode } from "react";
import * as THREE from "three";

const noMatchMediaSubscription = () => () => undefined;

function useMediaQuery(query: string): boolean {
  const mediaQuery = useMemo(
    () => (typeof window.matchMedia === "function" ? window.matchMedia(query) : null),
    [query],
  );
  const subscribe = useMemo(
    () =>
      mediaQuery
        ? (onChange: () => void) => {
            mediaQuery.addEventListener("change", onChange);
            return () => mediaQuery.removeEventListener("change", onChange);
          }
        : noMatchMediaSubscription,
    [mediaQuery],
  );
  return useSyncExternalStore(
    subscribe,
    () => mediaQuery?.matches ?? false,
    () => false,
  );
}

function createNetwork(count: number, radius: number) {
  const positions = new Float32Array(count * 3);
  const linePositions: number[] = [];

  for (let index = 0; index < count; index += 1) {
    const y = 1 - (index / (count - 1)) * 2;
    const radial = Math.sqrt(1 - y * y);
    const angle = Math.PI * (3 - Math.sqrt(5)) * index;
    const x = Math.cos(angle) * radial;
    const z = Math.sin(angle) * radial;
    positions.set([x * radius, y * radius, z * radius], index * 3);

  }

  for (let index = 0; index < count; index += 1) {
    const start = index * 3;
    const x = positions[start] ?? 0;
    const y = positions[start + 1] ?? 0;
    const z = positions[start + 2] ?? 0;
    for (const offset of [1, 8]) {
      const nextIndex = index + offset;
      if (nextIndex < count) {
        linePositions.push(
          x,
          y,
          z,
          positions[nextIndex * 3] ?? 0,
          positions[nextIndex * 3 + 1] ?? 0,
          positions[nextIndex * 3 + 2] ?? 0,
        );
      }
    }
  }

  return { positions, linePositions: new Float32Array(linePositions) };
}

function NetworkOrb({ compact }: { compact: boolean }) {
  const group = useRef<THREE.Group>(null);
  const pointer = useRef(new THREE.Vector2());
  const elapsed = useRef(0);
  const reducedMotion = useReducedMotion();
  const count = compact ? 72 : 128;
  const network = useMemo(() => createNetwork(count, 1.32), [count]);
  const pointsGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(network.positions, 3));
    return geometry;
  }, [network.positions]);
  const linesGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(network.linePositions, 3),
    );
    return geometry;
  }, [network.linePositions]);

  useFrame((_, delta) => {
    const current = group.current;
    if (!current) {
      return;
    }
    elapsed.current += delta;
    if (reducedMotion) {
      current.rotation.set(0, 0, 0);
      current.position.y = 0;
      return;
    }
    current.rotation.y += delta * 0.08;
    current.rotation.x = THREE.MathUtils.damp(
      current.rotation.x,
      pointer.current.y * 0.12,
      2.2,
      delta,
    );
    current.rotation.z = THREE.MathUtils.damp(
      current.rotation.z,
      pointer.current.x * -0.1,
      2.2,
      delta,
    );
    current.position.y = Math.sin(elapsed.current * 0.35) * 0.045;
  });

  return (
    <group
      onPointerMove={(event) => {
        pointer.current.set(
          (event.point.x / 2.2) * 0.5,
          (event.point.y / 1.8) * 0.5,
        );
      }}
      onPointerOut={() => pointer.current.set(0, 0)}
      ref={group}
    >
      <Float
        floatIntensity={reducedMotion ? 0 : 0.18}
        rotationIntensity={reducedMotion ? 0 : 0.08}
        speed={reducedMotion ? 0 : 0.4}
      >
        <mesh>
          <icosahedronGeometry args={[1.04, 2]} />
          <meshPhysicalMaterial
            color="#11253a"
            emissive="#0b3139"
            emissiveIntensity={0.48}
            metalness={0.52}
            roughness={0.38}
            transparent
            transmission={0.08}
            wireframe
            opacity={0.3}
          />
        </mesh>
      </Float>
      <lineSegments geometry={linesGeometry}>
        <lineBasicMaterial color="#58cfc9" transparent opacity={0.22} />
      </lineSegments>
      <points geometry={pointsGeometry}>
        <pointsMaterial
          color="#8ce8e3"
          size={compact ? 0.034 : 0.027}
          sizeAttenuation
          transparent
          opacity={0.86}
        />
      </points>
      <mesh>
        <sphereGeometry args={[0.12, 24, 24]} />
        <meshStandardMaterial
          color="#86f1e4"
          emissive="#32cbb3"
          emissiveIntensity={1.7}
          metalness={0.35}
          roughness={0.25}
        />
      </mesh>
      <mesh rotation={[0.8, 0.2, -0.35]}>
        <torusGeometry args={[1.56, 0.007, 6, 128]} />
        <meshBasicMaterial color="#67d9e6" transparent opacity={0.52} />
      </mesh>
      <mesh rotation={[-0.55, 0.45, 0.6]}>
        <torusGeometry args={[1.72, 0.004, 6, 128]} />
        <meshBasicMaterial color="#a68af8" transparent opacity={0.3} />
      </mesh>
    </group>
  );
}

function SceneFallback() {
  return (
    <div
      aria-label="A static illustration of an interconnected student data network"
      className="scene-static-fallback"
      role="img"
    >
      <div className="scene-fallback-orbit scene-fallback-orbit-one" />
      <div className="scene-fallback-orbit scene-fallback-orbit-two" />
      <div className="scene-fallback-core" />
      <span className="scene-fallback-node scene-fallback-node-one" />
      <span className="scene-fallback-node scene-fallback-node-two" />
      <span className="scene-fallback-node scene-fallback-node-three" />
      <span className="scene-fallback-node scene-fallback-node-four" />
      <span className="scene-fallback-caption">A connected view of learning</span>
    </div>
  );
}

class SceneErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    return this.state.hasError ? <SceneFallback /> : this.props.children;
  }
}

export function DecorativeDataScene() {
  const narrowViewport = useMediaQuery("(max-width: 640px)");
  const coarsePointer = useMediaQuery("(pointer: coarse)");
  const compact = narrowViewport || coarsePointer;
  const reducedMotion = useReducedMotion();

  return (
    <div aria-hidden="true" className="hero-scene-canvas">
      <SceneErrorBoundary>
        <Canvas
          camera={{ position: [0, 0, 4.5], fov: 38 }}
          dpr={compact ? [0.75, 1] : [1, 1.4]}
          fallback={<SceneFallback />}
          frameloop={reducedMotion ? "demand" : "always"}
          gl={{
            alpha: true,
            antialias: !compact,
            powerPreference: compact ? "low-power" : "default",
          }}
          onCreated={({ gl }) => {
            gl.setClearColor("#07111f", 0);
          }}
        >
          <ambientLight intensity={0.6} />
          <pointLight color="#50d6d0" intensity={22} position={[2.2, 2, 2.4]} />
          <pointLight color="#8b76eb" intensity={9} position={[-2.2, -1.4, -2]} />
          <Suspense fallback={null}>
            <NetworkOrb compact={compact} />
            {!compact ? (
              <Sparkles
                count={34}
                color="#9ae9e4"
                opacity={0.44}
                scale={4.2}
                size={1.3}
                speed={reducedMotion ? 0 : 0.1}
              />
            ) : null}
          </Suspense>
          <OrbitControls
            enablePan={false}
            enableRotate={!reducedMotion}
            enableZoom={false}
            maxPolarAngle={Math.PI / 1.8}
            minPolarAngle={Math.PI / 3}
            rotateSpeed={0.22}
          />
        </Canvas>
      </SceneErrorBoundary>
    </div>
  );
}
