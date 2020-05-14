/**********************************************************************************************************************
This file is part of the Control Toolbox (https://github.com/ethz-adrl/control-toolbox), copyright by ETH Zurich.
Licensed under the BSD-2 license (see LICENSE file in main directory)
**********************************************************************************************************************/

#pragma once

namespace ct {
namespace core {

//! class for a general switched discrete quadratic system or quadratized discrete system
/*!
 * Defines the interface for a switched discrete quadratized system
 *
 * \tparam STATE_DIM size of state vector
 * \tparam CONTROL_DIM size of input vector
 */
template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR = double>
class SwitchedDiscreteQuadraticSystem : public DiscreteQuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef typename std::shared_ptr<DiscreteQuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>> QuadraticSystemPtr;
    typedef Switched<QuadraticSystemPtr> SwitchedQuadraticSystems;

    typedef DiscreteControlledSystem<STATE_DIM, CONTROL_DIM, SCALAR> Base;
    typedef typename Base::time_t time_t;

    typedef typename Base::state_vector_t state_vector_t;
    typedef typename Base::control_vector_t control_vector_t;

    typedef StateMatrix<STATE_DIM, SCALAR> state_matrix_t;                              //!< state Jacobian type
    typedef StateControlMatrix<STATE_DIM, CONTROL_DIM, SCALAR> state_control_matrix_t;  //!< input Jacobian type

    //! default constructor
    /*!
     * @param type system type
     */
    SwitchedDiscreteQuadraticSystem(const SwitchedQuadraticSystems& switchedQuadraticSystems,
        const DiscreteModeSequence& discreteModeSequence,
        const SYSTEM_TYPE& type = SYSTEM_TYPE::GENERAL)
        : DiscreteQuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>(type),
          switchedQuadraticSystems_(switchedQuadraticSystems),
          discreteModeSequence_(discreteModeSequence){};

    //! copy constructor
    SwitchedDiscreteQuadraticSystem(const SwitchedDiscreteQuadraticSystem& arg)
        : DiscreteQuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>(arg), discreteModeSequence_(arg.discreteModeSequence_)
    {
        switchedQuadraticSystems_.clear();
        for (auto& subSystem : arg.switchedQuadraticSystems_)
        {
            switchedQuadraticSystems_.emplace_back(subSystem->clone());
        }
    };

    //! destructor
    virtual ~SwitchedDiscreteQuadraticSystem(){};

    //! deep cloning
    virtual SwitchedDiscreteQuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>* clone() const override
    {
        return new SwitchedDiscreteQuadraticSystem(*this);
    };

    //! compute the system dynamics
    /*!
     * This computes the system dynamics
     * \f[
     *  x_{n+1} = x_n^T fxx_{i} x_n + x_n^T fxu_{i} x_n + u_n^T fuu_{i} u_n + fx_{i} x_n + fu_{i} u_n
     * \f]
     * @param state current state
     * @param n current time index
     * @param control control input
     * @param stateNext propagated state
     */
    virtual void propagateControlledDynamics(const state_vector_t& state,
        const time_t n,
        const control_vector_t& control,
        state_vector_t& stateNext) override
    {
        auto mode = discreteModeSequence_.getPhaseFromTime(n);
        switchedQuadraticSystems_[mode]->propagateControlledDynamics(state, n, control, stateNext);
    };

    //! retrieve discrete-time quadratic system matrices fxx, fxu, fuu, fx, and fu for mode i, active at time n.
    /*!
     * This computes the system dynamics
     * \f[
     *  x_{n+1} = x_n^T fxx_{i} x_n + x_n^T fxu_{i} x_n + u_n^T fuu_{i} u_n + fx_{i} x_n + fu_{i} u_n
     * \f]
     *
     * Note that the inputs x_next and subSteps are potentially being ignored
     * for 'true' discrete system models but are relevant for sensitivity
     * calculations if the underlying system is continuous.
     *
     * @param x the state setpoint at n
     * @param u the control setpoint at n
     * @param n the time setpoint
     * @param x_next the state at n+1
     * @param subSteps number of substeps to use in sensitivity calculation
     * @param fxx the resulting quadratic system matrix fxx
     * @param fxu the resulting quadratic system matrix fxu
     * @param fuu the resulting quadratic system matrix fuu
     * @param fx the resulting quadratic system matrix fx
     * @param fu the resulting quadratic system matrix fu
     */
    virtual void getQuadratizedDynamics(const state_vector_t& x,
        const control_vector_t& u,
        const state_vector_t& x_next,
        const int n,
        size_t subSteps,
        state_matrix_t& fxx,
        state_matrix_t& fxu,
        state_control_matrix_t& fuu,
        state_matrix_t& fx,
        state_control_matrix_t& fu) override
    {
        auto mode = discreteModeSequence_.getPhaseFromTime(n);
        switchedQuadraticSystems_[mode]->getQuadratizedDynamics(x, u, x_next, n, subSteps, fxx, fxu, fuu, fx, fu);
    };

    using DiscreteQuadraticSystem<STATE_DIM, CONTROL_DIM, SCALAR>::getQuadratizedDynamics;

protected:
    SwitchedQuadraticSystems switchedQuadraticSystems_;  //!< Switched quadratic system container
    DiscreteModeSequence discreteModeSequence_;    //!< the prespecified mode sequence
};

}  // namespace core
}  // namespace ct