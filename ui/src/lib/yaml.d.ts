/** Type declarations for YAML files imported via @rollup/plugin-yaml. */
declare module '*.yaml' {
  const content: Record<string, unknown>;
  export default content;
}
