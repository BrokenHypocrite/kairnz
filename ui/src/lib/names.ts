/**
 * Display names loaded from config/names.yaml via the @rollup/plugin-yaml Vite plugin.
 * Content lives in YAML; this module only provides a typed accessor.
 */
import raw from '../../../config/names.yaml';

/** Typed shape of config/names.yaml. */
export interface Names {
  game: string;
  pieces: {
    stone: string;
    pillar: string;
    spire: string;
    keystone: string;
  };
  players: {
    P1: string;
    P2: string;
  };
  spire_modes: {
    Dragon: string;
    Queen: string;
  };
  draw_reasons: {
    MaxPlies: string;
    Repetition: string;
  };
  check_banner: string;
}

export const names = raw as unknown as Names;
