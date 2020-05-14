/**********************************************************************************************************************
This file is part of the Control Toolbox (https://github.com/ethz-adrl/control-toolbox), copyright by ETH Zurich.
Licensed under the BSD-2 license (see LICENSE file in main directory)
**********************************************************************************************************************/

#pragma once

namespace ct {
namespace core {

//! settings for the SensitivityApproximation
struct SensitivityQuadApproximationSettings
{
    //! different discrete-time approximations to quadratic systems
    enum class APPROXIMATION
    {
        FORWARD_EULER = 0,
        BACKWARD_EULER,
        SYMPLECTIC_EULER,
        TUSTIN,
        MATRIX_EXPONENTIAL
    };

    SensitivityQuadApproximationSettings(double dt, APPROXIMATION approx) : dt_(dt), approximation_(approx) {}
    //! discretization time-step
    double dt_;

    //! type of discretization strategy used.
    APPROXIMATION approximation_;
};

template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR = double>
class SensitivityQuad : public DiscreteQuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<StateVectorArray<STATE_DIM, SCALAR>> StateVectorArrayPtr;
    typedef std::shared_ptr<ControlVectorArray<CONTROL_DIM, SCALAR>> ControlVectorArrayPtr;


    SensitivityQuad() : xSubstep_(nullptr), uSubstep_(nullptr) {}
    virtual ~SensitivityQuad() {}
    virtual SensitivityQuad<STATE_DIM, CONTROL_DIM, SCALAR>* clone() const override
    {
        throw std::runtime_error("clone not implemented for SensitivityQuad");
    }

    virtual void setQuadraticSystem(const std::shared_ptr<QuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>>& quadraticSystem) = 0;

    //! update the time discretization
    virtual void setTimeDiscretization(const SCALAR& dt) = 0;

    //! update the approximation type for the discrete-time system
    virtual void setApproximation(const SensitivityQuadApproximationSettings::APPROXIMATION& approx) {}
    /*!
	 * Set the trajectory reference for quadratization. This should also include potential substeps that the integrator produces.
	 * @param x
	 * @param u
	 */
    void setSubstepTrajectoryReference(
        std::vector<StateVectorArrayPtr, Eigen::aligned_allocator<StateVectorArrayPtr>>* xSubstep,
        std::vector<ControlVectorArrayPtr, Eigen::aligned_allocator<ControlVectorArrayPtr>>* uSubstep)
    {
        xSubstep_ = xSubstep;
        uSubstep_ = uSubstep;
    }

    // TODO: Need to modify to ensure we the right types of all the matrices, for now just focus on software architecture
    //! retrieve discrete-time quadratic system matrices fxx, fxu, fuu, fx, fu.
    /*!
     * @param x	the state setpoint
     * @param u the control setpoint
     * @param n the time setpoint
     * @param numSteps number of timesteps of trajectory for which to get the sensitivity for
     * @param fxx the resulting quadratic system matrix fxx
     * @param fxu the resulting quadratic system matrix fxu
     * @param fuu the resulting quadratic system matrix fuu
     * @param fx the resulting quadratic system matrix fx
     * @param fu the resulting quadratic system matrix fu
     */
    virtual void getQuadratizedDynamics(const StateVector<STATE_DIM, SCALAR>& x,
        const ControlVector<CONTROL_DIM, SCALAR>& u,
        const StateVector<STATE_DIM, SCALAR>& x_next,
        const int n,
        size_t numSteps,
        StateMatrix<STATE_DIM, SCALAR>& fxx,
        StateMatrix<STATE_DIM, SCALAR>& fxu,
        StateControlMatrix<STATE_DIM, CONTROL_DIM, SCALAR>& fuu,
        StateMatrix<STATE_DIM, SCALAR>& fx,
        StateControlMatrix<STATE_DIM, CONTROL_DIM, SCALAR>& fu) override = 0;

protected:
    std::vector<StateVectorArrayPtr, Eigen::aligned_allocator<StateVectorArrayPtr>>* xSubstep_;
    std::vector<ControlVectorArrayPtr, Eigen::aligned_allocator<ControlVectorArrayPtr>>* uSubstep_;
};
}
}
