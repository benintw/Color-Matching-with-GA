import streamlit as st
from color_matching_ga import GAConfig, ColorMatchingGA, RealTimeVisualizer
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt


def run_ga_with_progress():
    # Get color from color picker
    color = st.color_picker("Choose target color", "#DCA521")
    # Convert hex to RGB
    target_rgb = np.array([int(color[1:][i : i + 2], 16) for i in (0, 2, 4)])

    col1, col2 = st.columns(2)
    with col1:
        population = st.slider("Population Size", 100, 2000, 1000)
        grid_size = st.slider("Grid Size", 2, 8, 4)
    with col2:
        max_gen = st.slider("Max Generations", 100, 2000, 1000)
        mutation_rate = st.slider("Mutation Rate", 0.001, 0.1, 0.01)

    if st.button("Start Evolution"):
        config = GAConfig(
            popsize=population,
            max_gen=max_gen,
            p_mutation=mutation_rate,
            chromosome_dim=(grid_size, grid_size),
            target_value=target_rgb,
        )

        ga = ColorMatchingGA(config)

        # Setup progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Setup plots
        fitness_plot = st.empty()
        color_display = st.empty()

        for gen in range(max_gen):
            # Evolution step
            fitness = ga._calc_fitness(ga.population)
            elite, elite_fitness = ga._update_statistics(fitness)

            # Update progress
            progress = gen / max_gen
            progress_bar.progress(progress)
            status_text.text(
                f"Generation {gen}/{max_gen} - Best Fitness: {elite_fitness:.2f}"
            )

            # Update visualizations periodically
            if gen % 10 == 0:
                # Plot fitness history
                fig_fitness = plt.figure(figsize=(10, 4))
                plt.plot(ga.best_fitness_values)
                plt.title("Fitness History")
                plt.xlabel("Generation")
                plt.ylabel("Fitness")
                fitness_plot.pyplot(fig_fitness)
                plt.close()

                # Display current best pattern
                fig_pattern = plt.figure(figsize=(5, 5))
                plt.imshow(elite / 255.0)
                plt.title(f"Best Pattern (Gen {gen})")
                color_display.pyplot(fig_pattern)
                plt.axis("off")
                plt.close()

            # Evolution step
            if ga._should_stop(elite_fitness, ga.generation_without_improvement):
                break

            mating_pool = ga._generate_mating_pool(fitness)
            new_population = ga._create_offspring(mating_pool)
            ga.population = np.vstack((new_population, np.expand_dims(elite, 0)))

        st.success("Evolution completed!")

        # Save final animation
        buffer = io.BytesIO()
        RealTimeVisualizer.create_animation(
            target_rgb, np.array(ga.best_chromosomes), grid_size
        )
        st.download_button(
            label="Download Evolution Animation",
            data=buffer,
            file_name="evolution.gif",
            mime="image/gif",
        )


if __name__ == "__main__":
    st.title("Color Pattern Evolution Demo")
    st.write(
        """
    This demo shows how genetic algorithms can evolve patterns to match a target color.
    Choose your parameters and watch the evolution happen!
    """
    )

    run_ga_with_progress()
